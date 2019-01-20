/// Working towards a "Hello world" for the Looking glass - the idea is to render a 3D spinning cube

use gl;
extern crate glutin;
use cgmath;
use cgmath::{Angle, Point3, Matrix, Matrix4, Rad, Vector3, Vector4};

use glutin::GlContext;

use std::io::{self, Write};
use std::ffi::CStr;
use std::mem;
use std::ptr;
use std::time::Instant;

struct GlState {
    vb : u32,
    program: u32,
}


// Next steps:
//   * Figure out how to make fixed, textured, squares in right orientation for the mpv shader
//   * Render to FBOs
//   * Render multiple perspectives of same scene
//   * Incorporate mpv shader
fn main() {
    let mut events_loop = glutin::EventsLoop::new();

    // enumerate monitors
    let monitor = {
        for (num, monitor) in events_loop.get_available_monitors().enumerate() {
            println!("Monitor #{}: {:?}", num, monitor.get_name());
        }

        print!("Please write the number of the monitor to use: ");
        io::stdout().flush().unwrap();

        let mut num = String::new();
        io::stdin().read_line(&mut num).unwrap();
        let num = num.trim().parse().ok().expect("Please enter a number");
        let monitor = events_loop.get_available_monitors().nth(num).expect("Please enter a valid ID");

        println!("Using {:?}", monitor.get_name());

        monitor
    };

    let window = glutin::WindowBuilder::new()
        .with_title("Hello, world!")
        .with_fullscreen(Some(monitor));
    let context = glutin::ContextBuilder::new()
        .with_vsync(true);
    let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();

    unsafe {
        gl_window.make_current().unwrap();
    }

    gl::load_with(|ptr| gl_window.context().get_proc_address(ptr) as *const _);

    print_opengl_version();

    let gl_state = gl_setup();

    let view = Matrix4::look_at(Point3::new(0.5, 0.5, 0.5), // eye
                                Point3::new(0.0, 0.0, 0.0), // center
                                Vector3::new(0.0, 1.0, 0.0)); // up
    load_uniform_matrix(&gl_state, "view".to_string(), view);

    let proj = perspective_matrix(Rad(1.0), 3840.0/2160.0, 0.1, 100.0);
    load_uniform_matrix(&gl_state, "projection".to_string(), proj);

    let mut running = true;
    let speed = Rad(2.0); // rad/sec
    let start = Instant::now();

    while running {
        events_loop.poll_events(|event| {
            match event {
                glutin::Event::WindowEvent{ event, .. } => match event {
                    glutin::WindowEvent::CloseRequested => running = false,
                    glutin::WindowEvent::Resized(logical_size) => {
                        let dpi_factor = gl_window.get_hidpi_factor();
                        gl_window.resize(logical_size.to_physical(dpi_factor));
                    },
                    glutin::WindowEvent::KeyboardInput { input, .. } => {
                        match input.virtual_keycode {
                            Some(glutin::VirtualKeyCode::Escape) => running = false,
                            _ => (),
                        }
                    },
                    _ => ()
                },
                _ => ()
            }
        });

        let duration = Instant::now().duration_since(start);
        let duration_f32 = duration.as_secs() as f32 + duration.subsec_nanos() as f32 * 1e-9;
        let theta = (speed * duration_f32).normalize();

        unsafe {
            let vert_data = generate_cube(Matrix4::<f32>::from_angle_z(theta));
            gl::BindBuffer(gl::ARRAY_BUFFER, gl_state.vb);
            gl::BufferData(gl::ARRAY_BUFFER,
                               (vert_data.len() * mem::size_of::<f32>()) as gl::types::GLsizeiptr,
                               vert_data.as_ptr() as *const _, gl::STREAM_DRAW);

            let pos_attrib = gl::GetAttribLocation(gl_state.program, b"position\0".as_ptr() as *const _);
            let color_attrib = gl::GetAttribLocation(gl_state.program, b"color\0".as_ptr() as *const _);
            gl::VertexAttribPointer(pos_attrib as gl::types::GLuint, 3, gl::FLOAT, 0,
                                        6 * mem::size_of::<f32>() as gl::types::GLsizei,
                                        ptr::null());
            gl::VertexAttribPointer(color_attrib as gl::types::GLuint, 3, gl::FLOAT, 0,
                                        6 * mem::size_of::<f32>() as gl::types::GLsizei,
                                        (3 * mem::size_of::<f32>()) as *const () as *const _);
            gl::EnableVertexAttribArray(pos_attrib as gl::types::GLuint);
            gl::EnableVertexAttribArray(color_attrib as gl::types::GLuint);

            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::DrawArrays(gl::TRIANGLES, 0, vert_data.len() as i32 / 5);
        }

        gl_window.swap_buffers().unwrap();
    }
}

/// http://davidlively.com/programming/graphics/opengl-matrices/perspective-projection/
fn perspective_matrix(theta: Rad<f32>, aspect: f32, near: f32, far:f32) -> Matrix4<f32>{
    let hh = near * (theta / 2.0).tan();
    let hw = hh * aspect;
    let depth = far - near;

    Matrix4::new(
        near/hw, 0.0,     0.0,                  0.0,
        0.0,     near/hh, 0.0,                  0.0,
        0.0,     0.0,     -(far+near)/depth,   -1.0,
        0.0,     0.0,     -2.0*far*near/depth,  0.0,
    )
}

/// Loads a matrix from the main program in to a "uniform" shader matrix
fn load_uniform_matrix(gl_state: &GlState, mut dest: String, m: Matrix4<f32>) {
    dest.push('\0');
    unsafe {
        let matrix_id = gl::GetUniformLocation(gl_state.program, dest.as_ptr() as *const _);
        gl::UniformMatrix4fv(matrix_id, 1, false as gl::types::GLboolean, m.as_ptr() as *const _);
    }
}

/// Must be called after gl::load_with()
fn print_opengl_version() {
    let version = unsafe {
        let data = CStr::from_ptr(gl::GetString(gl::VERSION) as *const _).to_bytes().to_vec();
        String::from_utf8(data).unwrap()
    };
    println!("OpenGL version {}", version);
}

fn gl_setup() -> GlState {
    let mut vb;
    let program;

    unsafe {
        gl::ClearColor(0.40, 0.10, 0.10, 1.0);

        let vs = gl::CreateShader(gl::VERTEX_SHADER);
        gl::ShaderSource(vs, 1, [VS_SRC.as_ptr() as *const _].as_ptr(), ptr::null());
        gl::CompileShader(vs);

        let fs = gl::CreateShader(gl::FRAGMENT_SHADER);
        gl::ShaderSource(fs, 1, [FS_SRC.as_ptr() as *const _].as_ptr(), ptr::null());
        gl::CompileShader(fs);

        program = gl::CreateProgram();
        gl::AttachShader(program, vs);
        gl::AttachShader(program, fs);
        gl::LinkProgram(program);
        gl::UseProgram(program);

        vb = mem::uninitialized();
        gl::GenBuffers(1, &mut vb);

        if gl::BindVertexArray::is_loaded() {
            let mut vao = mem::uninitialized();
            gl::GenVertexArrays(1, &mut vao);
            gl::BindVertexArray(vao);
        }

        // gl::Enable(gl::CULL_FACE);
        // gl::CullFace(gl::BACK);
        gl::Enable(gl::DEPTH_TEST);
    }

    GlState {
        vb,
        program,
    }
}

/// Makes a multicoloured cube 0.25 on a side, at origin transformed by m
fn generate_cube(m: Matrix4<f32>) -> Vec<f32> {
    fn square(out: &mut Vec<f32>, v: &[Vector4<f32>], color: &[f32]) {
        out.extend_from_slice(&[v[0].x, v[0].y, v[0].z, color[0], color[1], color[2]]);
        out.extend_from_slice(&[v[1].x, v[1].y, v[1].z, color[0], color[1], color[2]]);
        out.extend_from_slice(&[v[2].x, v[2].y, v[2].z, color[0], color[1], color[2]]);

        out.extend_from_slice(&[v[2].x, v[2].y, v[2].z, color[0], color[1], color[2]]);
        out.extend_from_slice(&[v[3].x, v[3].y, v[3].z, color[0], color[1], color[2]]);
        out.extend_from_slice(&[v[0].x, v[0].y, v[0].z, color[0], color[1], color[2]]);
    }

    // Vector4 here, just because multiplying by 4x4 matrix
    let va = m * Vector4::new(-0.125, 0.125, -0.125, 1.0);
    let vb = m * Vector4::new(0.125, 0.125, -0.125, 1.0);
    let vc = m * Vector4::new(0.125, -0.125, -0.125, 1.0);
    let vd = m * Vector4::new(-0.125, -0.125, -0.125, 1.0);

    let ve = m * Vector4::new(-0.125, 0.125, 0.125, 1.0);
    let vf = m * Vector4::new(0.125, 0.125, 0.125, 1.0);
    let vg = m * Vector4::new(0.125, -0.125, 0.125, 1.0);
    let vh = m * Vector4::new(-0.125, -0.125, 0.125, 1.0);

    // 6 faces, 2 triangles each, 3 verts/tri, 7 f32s per vert
    let mut ret = Vec::with_capacity(6 * 2 * 3 * 7);
    square(&mut ret, &[va, vb, vc, vd], &[1.0, 0.0, 0.0]);
    square(&mut ret, &[vb, vf, vg, vc], &[1.0, 1.0, 0.0]);
    square(&mut ret, &[vf, ve, vh, vg], &[0.0, 1.0, 0.0]);
    square(&mut ret, &[ve, va, vd, vh], &[0.0, 1.0, 1.0]);
    square(&mut ret, &[ve, vf, vb, va], &[0.0, 0.0, 1.0]);
    square(&mut ret, &[vd, vc, vg, vh], &[1.0, 0.0, 1.0]);
    ret
}

// Vertex shader
const VS_SRC: &'static [u8] = b"
#version 100
precision mediump float;

attribute vec3 position;
attribute vec3 color;

uniform mat4 view;
uniform mat4 projection;

varying vec3 v_color;

void main() {
    gl_Position = projection * view * vec4(position, 1.0);
    v_color = color;
}
\0";

// Fragment shader
const FS_SRC: &'static [u8] = b"
#version 100
precision mediump float;

varying vec3 v_color;

void main() {
    gl_FragColor = vec4(v_color, 1.0);
}
\0";
