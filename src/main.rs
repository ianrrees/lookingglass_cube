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
    framebuffer : u32,
    vb : u32,
    program: u32,
    vao: u32,
}

// Next steps:
//   * Render to FBOs
//   * Render multiple perspectives of same scene
//   * Update mpv shader
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

    unsafe {
        gl::ClearColor(0.40, 0.10, 0.10, 1.0);

        // gl::Enable(gl::CULL_FACE);
        // gl::CullFace(gl::BACK);
        gl::Enable(gl::DEPTH_TEST);
    }

    let main_state = setup_main();
    let quilt_state = setup_quilt();
    use_gl_state(&main_state);

    let view = Matrix4::look_at(Point3::new(0.5, 0.5, 0.5), // eye
                                Point3::new(0.0, 0.0, 0.0), // center
                                Vector3::new(0.0, 1.0, 0.0)); // up
    load_uniform_matrix(&main_state, "view".to_string(), view);

    // TODO get these from looking glass or whatever
    let screen_width = 3840;
    let screen_height = 2160;

    let proj = perspective_matrix(Rad(1.0), screen_width as f32 / screen_height as f32, 0.1, 100.0);
    load_uniform_matrix(&main_state, "projection".to_string(), proj);

    // I assume that we want an 8:5 ratio, like the overall display, with roughly 91kpx
    let quilt_width = 320;
    let quilt_height = 200;

    let quilt_cols = 5;
    let quilt_rows = 9;
    let quilt_verts = generate_quilt(quilt_rows, quilt_cols);

    let mut tex;
    unsafe {
        tex = mem::uninitialized();
        gl::GenTextures(1, &mut tex);
        gl::ActiveTexture(gl::TEXTURE0);
        gl::BindTexture(gl::TEXTURE_2D_ARRAY, tex);

        gl::TexStorage3D(gl::TEXTURE_2D_ARRAY, 1, gl::RGB8,
            quilt_width,
            quilt_height,
            quilt_rows as i32 * quilt_cols as i32 // Array count
            );

        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32); // TODO use OpenGL type instead of i32
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
    }

    let mut running = true;
    let speed = Rad(2.0); // rad/sec
    let start = Instant::now();


    let mut quilt_depth_buffer;
    unsafe {
        // Set up a depth buffer, which we'll re-use between framebuffers
        quilt_depth_buffer = mem::uninitialized();
        gl::GenRenderbuffers(1, &mut quilt_depth_buffer);
        gl::BindRenderbuffer(gl::RENDERBUFFER, quilt_depth_buffer);
        gl::RenderbufferStorage(gl::RENDERBUFFER, gl::DEPTH_COMPONENT, quilt_width, quilt_height);
        gl::FramebufferRenderbuffer(gl::FRAMEBUFFER, gl::DEPTH_ATTACHMENT,
                                    gl::RENDERBUFFER, quilt_depth_buffer);
    }

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

        use_gl_state(&main_state);

        unsafe {
            // Framebuffer stuff initially from https://www.opengl-tutorial.org/intermediate-tutorials/tutorial-14-render-to-texture/
            gl::BindRenderbuffer(gl::RENDERBUFFER, quilt_depth_buffer);
            gl::FramebufferRenderbuffer(gl::FRAMEBUFFER, gl::DEPTH_ATTACHMENT,
                                        gl::RENDERBUFFER, quilt_depth_buffer);

            // Seems to not be needed?
            // Set the list of draw buffers.
            let draw_buffer = gl::COLOR_ATTACHMENT0;
            gl::DrawBuffers(1, &draw_buffer as *const u32); // "1" is the size of DrawBuffers

            let vert_data = generate_cube(Matrix4::<f32>::from_angle_z(theta));
            gl::BufferData(gl::ARRAY_BUFFER,
                (vert_data.len() * mem::size_of::<f32>()) as gl::types::GLsizeiptr,
                vert_data.as_ptr() as *const _, gl::STREAM_DRAW);

            let pos_attrib = gl::GetAttribLocation(main_state.program, b"position\0".as_ptr() as *const _);
            let color_attrib = gl::GetAttribLocation(main_state.program, b"color\0".as_ptr() as *const _);
            gl::VertexAttribPointer(pos_attrib as gl::types::GLuint, 3, gl::FLOAT, 0,
                6 * mem::size_of::<f32>() as gl::types::GLsizei, ptr::null());
            gl::VertexAttribPointer(color_attrib as gl::types::GLuint, 3, gl::FLOAT, 0,
                6 * mem::size_of::<f32>() as gl::types::GLsizei,
                (3 * mem::size_of::<f32>()) as *const () as *const _);
            gl::EnableVertexAttribArray(pos_attrib as gl::types::GLuint);
            gl::EnableVertexAttribArray(color_attrib as gl::types::GLuint);

            gl::Viewport(0, 0, quilt_width, quilt_height);

            for frame in 0..45 {
                // Last two are mip level and layer
                gl::FramebufferTextureLayer(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, tex, 0, frame);

                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
                gl::DrawArrays(gl::TRIANGLES, 0, vert_data.len() as i32 / 5);
            }
        }


        use_gl_state(&quilt_state);
        unsafe {
            // Switch back to drawing on the screen
            gl::Viewport(0, 0, screen_width, screen_height); // Render on the whole framebuffer, complete from the lower left corner to the upper right

            // quilt_verts contains X, Y, tex_x, tex_y
            gl::BufferData(gl::ARRAY_BUFFER,
                (quilt_verts.len() * mem::size_of::<f32>()) as gl::types::GLsizeiptr,
                quilt_verts.as_ptr() as *const _, gl::STREAM_DRAW); // TODO use the less volatile type

            let pos_attrib = gl::GetAttribLocation(quilt_state.program, b"position\0".as_ptr() as *const _);
            let tex_attrib = gl::GetAttribLocation(quilt_state.program, b"texcoord\0".as_ptr() as *const _);

            gl::VertexAttribPointer(pos_attrib as gl::types::GLuint, 2, gl::FLOAT, 0,
                5 * mem::size_of::<f32>() as gl::types::GLsizei, ptr::null());
            gl::VertexAttribPointer(tex_attrib as gl::types::GLuint, 3, gl::FLOAT, 0,
                5 * mem::size_of::<f32>() as gl::types::GLsizei,
                (2 * mem::size_of::<f32>()) as *const () as *const _);

            gl::EnableVertexAttribArray(pos_attrib as gl::types::GLuint);
            gl::EnableVertexAttribArray(tex_attrib as gl::types::GLuint);

            gl::Clear(gl::DEPTH_BUFFER_BIT);
            gl::DrawArrays(gl::TRIANGLES, 0, quilt_verts.len() as i32 / 5);
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

fn use_gl_state(state: &GlState) {
    unsafe {
        gl::BindFramebuffer(gl::FRAMEBUFFER, state.framebuffer);
        gl::UseProgram(state.program);
        gl::BindBuffer(gl::ARRAY_BUFFER, state.vb);
        gl::BindVertexArray(state.vao);
    }
}

/// Sets up the OpenGL program for rendering a grid of textured 2D rects
fn setup_quilt() -> GlState {
    const VERTEX_SHADER: &'static [u8] = b"
    #version 330
    precision mediump float;

    in vec2 position;
    in vec3 texcoord;

    out vec3 v_texcoord;

    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        v_texcoord = texcoord;
    }
    \0";

    // TODO The algo borrowed from lonetech relies on the texture coordinate varying from 0-1 across
    // the whole screen, but currently it's going from 0-1 across each quilt square...
    const FRAGMENT_SHADER: &'static [u8] = b"
    #version 330
    precision mediump float;

    uniform sampler2DArray quilt_sampler;

    in vec3 v_texcoord;

    out vec4 fragColor;

    // For a Standard Looking Glass
    const int width = 2560;
    const int height = 1600;
    const int dpi = 338;

    // Copy these calibration values from the EEPROM in your looking glass
    // TODO parameterise these
    const float slope = 5.044347763061523;
    const float center = 0.176902174949646;
    const float pitch = 49.81804275512695;

    const float tilt = -height / (width*slope);
    const float pitch_adjusted = pitch * width / dpi * cos(atan(1.0, slope));
    const float subp = 1.0 / (3*width) * pitch_adjusted;

    // TODO parameterise this
    const vec2 tiles = vec2(5,9);

    vec2 quilt_map(vec2 pos, float a) {
      // Y major positive direction, X minor negative direction
      vec2 tile = vec2(tiles.x-1,0), dir=vec2(-1,1);
      a = fract(a)*tiles.y;
      tile.y += dir.y*floor(a);
      a = fract(a)*tiles.x;
      tile.x += dir.x*floor(a);
      return (tile+pos)/tiles;
    }

    void main() {
      float a;
      a = (v_texcoord.x + v_texcoord.y*tilt)*pitch_adjusted - center;

      // gl_FragColor.x = texture(quilt_sampler, quilt_map(v_texcoord, a)).x;
      // gl_FragColor.y = texture(quilt_sampler, quilt_map(v_texcoord, a+subp)).y;
      // gl_FragColor.z = texture(quilt_sampler, quilt_map(v_texcoord, a+2*subp)).z;
      // gl_FragColor.w = 1.0;

      fragColor = texture(quilt_sampler, v_texcoord);
    }
    \0";

    let program;
    let mut vb;
    let mut vao;
    unsafe {
        let vs = gl::CreateShader(gl::VERTEX_SHADER);
        gl::ShaderSource(vs, 1, [VERTEX_SHADER.as_ptr() as *const _].as_ptr(), ptr::null());
        gl::CompileShader(vs);

        let fs = gl::CreateShader(gl::FRAGMENT_SHADER);
        gl::ShaderSource(fs, 1, [FRAGMENT_SHADER.as_ptr() as *const _].as_ptr(), ptr::null());
        gl::CompileShader(fs);

        program = gl::CreateProgram();
        gl::AttachShader(program, vs);
        gl::AttachShader(program, fs);
        gl::LinkProgram(program);

        vb = mem::uninitialized();
        gl::GenBuffers(1, &mut vb);

        vao = mem::uninitialized();
        gl::GenVertexArrays(1, &mut vao);
    }

    GlState{
        framebuffer: 0, // Render the quilts to the screen
        vb,
        program,
        vao,
    }
}

/// Returns program, vertex buffer tuple
fn setup_main() -> GlState {
    // Vertex shader
    const VERTEX_SHADER: &'static [u8] = b"
    #version 330
    precision mediump float;

    in vec3 position;
    in vec3 color;

    uniform mat4 view;
    uniform mat4 projection;

    out vec3 v_color;

    void main() {
        gl_Position = projection * view * vec4(position, 1.0);
        v_color = color;
    }
    \0";

    // Fragment shader
    const FRAGMENT_SHADER: &'static [u8] = b"
    #version 330
    precision mediump float;

    in vec3 v_color;
    out vec4 fragColor;

    void main() {
        gl_FragColor = vec4(v_color, 1.0);
    }
    \0";

    let mut framebuffer;
    let program;
    let mut vb;
    let mut vao;
    unsafe {
        framebuffer = mem::uninitialized();
        gl::GenFramebuffers(1, &mut framebuffer);

        let vs = gl::CreateShader(gl::VERTEX_SHADER);
        gl::ShaderSource(vs, 1, [VERTEX_SHADER.as_ptr() as *const _].as_ptr(), ptr::null());
        gl::CompileShader(vs);

        let fs = gl::CreateShader(gl::FRAGMENT_SHADER);
        gl::ShaderSource(fs, 1, [FRAGMENT_SHADER.as_ptr() as *const _].as_ptr(), ptr::null());
        gl::CompileShader(fs);

        program = gl::CreateProgram();
        gl::AttachShader(program, vs);
        gl::AttachShader(program, fs);
        gl::LinkProgram(program);

        vb = mem::uninitialized();
        gl::GenBuffers(1, &mut vb);

        vao = mem::uninitialized();
        gl::GenVertexArrays(1, &mut vao);
    }

    GlState{
        framebuffer, // Render the scene to a framebuffer bound to the appropriate texture
        vb,
        program,
        vao,
    }
}


// Makes the textured 2D quilt
fn generate_quilt(rows: u32, cols: u32) -> Vec<f32> {
    let mut ret = Vec::new();

    let half_width = 1.0 / cols as f32;
    let half_height = 1.0 / rows as f32;

    let mut frame = 0.0;
    'row_loop:
    for row in 0..rows {
        for column in 0..cols {
            // Compute centre of quilt "square"
            let x = -1.0 + (column * 2 + 1) as f32 * half_width;
            let y = -1.0 + (row * 2 + 1) as f32 * half_height;

            ret.extend_from_slice(&[x - half_width, y + half_height, 0.0, 0.0, frame]);
            ret.extend_from_slice(&[x + half_width, y + half_height, 1.0, 0.0, frame]);
            ret.extend_from_slice(&[x + half_width, y - half_height, 1.0, 1.0, frame]);

            ret.extend_from_slice(&[x - half_width, y + half_height, 0.0, 0.0, frame]);
            ret.extend_from_slice(&[x + half_width, y - half_height, 1.0, 1.0, frame]);
            ret.extend_from_slice(&[x - half_width, y - half_height, 0.0, 1.0, frame]);

            frame += 1.0;
        }
    }

    ret
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
