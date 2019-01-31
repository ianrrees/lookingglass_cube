/// A "Hello world" for the Looking glass - the idea is to render a 3D spinning cube

// The Looking Glass requires 45 different views, varying over 40 degrees of azimuth (left to right)
// which are interlaced together according to the calibration of each individual Looking Glass.  We
// accomplish both the scene rendering and the interlacing with OpenGL; the basic idea is to render
// the same scene 45 different times in to layers of a 2D texture array, then render a fullscreen
// rectangle through a vertex shader that interlaces those 45 layers.

use gl;
extern crate glutin;
use cgmath;
use cgmath::{Angle, Deg, EuclideanSpace, InnerSpace, Matrix, Matrix4, Point3, Rad, Vector3, Vector4};
use pluton::LookingGlass;
use glutin::GlContext;
use std::io::{self, Write};
use std::mem;
use std::ptr;
use std::time::Instant;

fn main() {
    let mut events_loop = glutin::EventsLoop::new();

    // Find the Looking Glass
    let glass;
    match find_looking_glass() {
        Some(lg) => {
            glass = lg;
            println!("Found looking glass s/n:{}", glass.serial);
        },
        None => {
            println!("Couldn't identify Looking Glass to display the cube on.");
            return;
        }
    }

    // Pick a monitor.  On my machine, the dimensions (and name, actually ;) ) don't match the ones
    // shown by "monitor" in the loop below.  Unsure why that is...
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
    let context = glutin::ContextBuilder::new().with_vsync(true);
    let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();

    unsafe {
        gl_window.make_current().unwrap();
    }

    gl::load_with(|ptr| gl_window.context().get_proc_address(ptr) as *const _);

    unsafe {
        gl::ClearColor(0.40, 0.10, 0.10, 1.0);
        gl::CullFace(gl::BACK);
        gl::Enable(gl::CULL_FACE);
        gl::Enable(gl::DEPTH_TEST);
    }

    let main_state = setup_main();
    let blend_state = setup_blend(&glass);
    use_gl_state(&main_state);

    let proj = perspective_matrix(Rad(1.0), // Field of view
                                  glass.screen_w as f32 / glass.screen_h as f32, // Aspect ratio
                                  0.1, 100.0); // Near and far planes
    load_uniform_matrix(&main_state, "projection".to_string(), proj);

    // I assume that we want an 8:5 ratio, like the overall display, with roughly 91kpx
    let view_width = 320;
    let view_height = 200;

    let view_count = 45;

    // Want to use the interlacing fragment shader over the whole display
    const VIEW_VERTS: [f32; 24] = [
        -1.0, 1.0,    0.0, 0.0,
        1.0, -1.0,    1.0, 1.0,
        1.0, 1.0,     1.0, 0.0,
        -1.0, 1.0,    0.0, 0.0,
        -1.0, -1.0,   0.0, 1.0,
        1.0, -1.0,    1.0, 1.0];

    let mut tex;
    unsafe {
        tex = mem::uninitialized();
        gl::GenTextures(1, &mut tex);
        gl::ActiveTexture(gl::TEXTURE0);
        gl::BindTexture(gl::TEXTURE_2D_ARRAY, tex);

        gl::TexStorage3D(gl::TEXTURE_2D_ARRAY, 1, gl::RGB8, view_width, view_height, view_count);

        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32); // TODO use OpenGL type instead of i32
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
    }

    let mut running = true;
    let speed = Rad(1.5); // rad/sec
    let start = Instant::now();

    let mut blend_depth_buffer;
    unsafe {
        // Set up a depth buffer, which we'll re-use between framebuffers
        blend_depth_buffer = mem::uninitialized();
        gl::GenRenderbuffers(1, &mut blend_depth_buffer);
        gl::BindRenderbuffer(gl::RENDERBUFFER, blend_depth_buffer);
        gl::RenderbufferStorage(gl::RENDERBUFFER, gl::DEPTH_COMPONENT, view_width, view_height);
        gl::FramebufferRenderbuffer(gl::FRAMEBUFFER, gl::DEPTH_ATTACHMENT,
                                    gl::RENDERBUFFER, blend_depth_buffer);
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

        // For cube spinnery
        let duration = Instant::now().duration_since(start);
        let duration_f32 = duration.as_secs() as f32 + duration.subsec_nanos() as f32 * 1e-9;
        let theta = (speed * duration_f32).normalize();

        // First, set up to render the 3D scene (cube, in our case) in to a 2D texture array
        use_gl_state(&main_state);
        unsafe {
            // Framebuffer stuff initially from https://www.opengl-tutorial.org/intermediate-tutorials/tutorial-14-render-to-texture/
            gl::BindRenderbuffer(gl::RENDERBUFFER, blend_depth_buffer);
            gl::FramebufferRenderbuffer(gl::FRAMEBUFFER, gl::DEPTH_ATTACHMENT,
                                        gl::RENDERBUFFER, blend_depth_buffer);

            // Seems to not be needed?
            // Set the list of draw buffers.
            let draw_buffer = gl::COLOR_ATTACHMENT0;
            gl::DrawBuffers(1, &draw_buffer as *const u32); // "1" is the size of DrawBuffers

            let pos_attrib = gl::GetAttribLocation(main_state.program, b"position\0".as_ptr() as *const _);
            let color_attrib = gl::GetAttribLocation(main_state.program, b"color\0".as_ptr() as *const _);
            gl::VertexAttribPointer(pos_attrib as gl::types::GLuint, 3, gl::FLOAT, 0,
                6 * mem::size_of::<f32>() as gl::types::GLsizei, ptr::null());
            gl::VertexAttribPointer(color_attrib as gl::types::GLuint, 3, gl::FLOAT, 0,
                6 * mem::size_of::<f32>() as gl::types::GLsizei,
                (3 * mem::size_of::<f32>()) as *const () as *const _);
            gl::EnableVertexAttribArray(pos_attrib as gl::types::GLuint);
            gl::EnableVertexAttribArray(color_attrib as gl::types::GLuint);

            gl::Viewport(0, 0, view_width, view_height);

            // Now that OpenGL is setup, draw the cube once per frame
            let cube_mat = Matrix4::<f32>::from_angle_z(theta);
            let vert_data = generate_cube(cube_mat);
            gl::BufferData(gl::ARRAY_BUFFER,
                (vert_data.len() * mem::size_of::<f32>()) as gl::types::GLsizeiptr,
                vert_data.as_ptr() as *const _, gl::STREAM_DRAW);

            // Render the same scene 45 times, in to the 45 layers of the texture buffer
            for view_num in 0..45 {
                // Last two arguments are mip level and layer
                gl::FramebufferTextureLayer(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, tex, 0, view_num);

                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                let view_matrix = looking_glass_lookat(Point3::new(0.5, 0.5, 1.0), // eye
                                                       Point3::new(0.0, 0.0, 0.0), // center
                                                       Vector3::new(0.0, 1.0, 0.0), // up
                                                       view_num); // which view
                load_uniform_matrix(&main_state, "view".to_string(), view_matrix);

                // /6 because each vertex from the cube data has 3 axes of position and colour
                gl::DrawArrays(gl::TRIANGLES, 0, vert_data.len() as i32 / 6);
            }
        }

        // Now, render a full-screen rectangle, interleaving the 2D texture array across it
        use_gl_state(&blend_state);
        unsafe {
            // Switch back to drawing on the screen
            gl::Viewport(0, 0, glass.screen_w as i32, glass.screen_h as i32);

            // VIEW_VERTS contains X, Y, tex_x, tex_y
            gl::BufferData(gl::ARRAY_BUFFER,
                (VIEW_VERTS.len() * mem::size_of::<f32>()) as gl::types::GLsizeiptr,
                VIEW_VERTS.as_ptr() as *const _, gl::STATIC_DRAW);

            let pos_attrib = gl::GetAttribLocation(blend_state.program, b"position\0".as_ptr() as *const _);
            let tex_attrib = gl::GetAttribLocation(blend_state.program, b"texcoord\0".as_ptr() as *const _);

            gl::VertexAttribPointer(pos_attrib as gl::types::GLuint, 2, gl::FLOAT, 0,
                4 * mem::size_of::<f32>() as gl::types::GLsizei, ptr::null());
            gl::VertexAttribPointer(tex_attrib as gl::types::GLuint, 2, gl::FLOAT, 0,
                4 * mem::size_of::<f32>() as gl::types::GLsizei,
                (2 * mem::size_of::<f32>()) as *const () as *const _);

            gl::EnableVertexAttribArray(pos_attrib as gl::types::GLuint);
            gl::EnableVertexAttribArray(tex_attrib as gl::types::GLuint);

            gl::Clear(gl::DEPTH_BUFFER_BIT);
            gl::DrawArrays(gl::TRIANGLES, 0, VIEW_VERTS.len() as i32 / 4);
        }

        gl_window.swap_buffers().unwrap();
    }
}

struct GlState {
    framebuffer : u32,
    vb : u32,
    program: u32,
    vao: u32,
}

fn find_looking_glass() -> Option<LookingGlass> {
    let mut glass : Option<LookingGlass> = None;

    for candidate in LookingGlass::findall() {
        match candidate {
            Ok(lg) => {
                if glass.is_none() {
                    glass = Some(lg);
                } else {
                    println!("Found multiple Looking Glasses, but can only handle one.");
                    return None;
                }
            },
            Err(e) => {
                println!("Error finding Looking Glass: {:?}", e);
            }
        }
    }
    glass
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

/// Creates a new eye point, which looks at center, but is rotated appropriately for view_num.
fn looking_glass_lookat(eye: Point3<f32>, center: Point3<f32>, up: Vector3<f32>,
                        view_num: i32) -> Matrix4<f32> {
    // Construct a vector that's coplanar with center-eye and up, and perpindicular to eye-center.
    let center_eye = eye - center;
    let left = center_eye.cross(up + center.to_vec());
    let axis = left.cross(center_eye).normalize();

    // Then, rotate center-eye around that vector according to the view_num.
    const NUM_VIEWS: i32 = 45;
    const CENTRE_VIEW: i32 = NUM_VIEWS / 2;
    const SPAN_PER_VIEW: f32 = 40.0 / NUM_VIEWS as f32; // degrees
    let rotation = Matrix4::<f32>::from_axis_angle(axis,
        Deg((CENTRE_VIEW - view_num) as f32 * SPAN_PER_VIEW));

    let new_eye = (rotation * center_eye.extend(1.0)).truncate() + center.to_vec();

    Matrix4::look_at(Point3::from_vec(new_eye), center, axis)
}

fn use_gl_state(state: &GlState) {
    unsafe {
        gl::BindFramebuffer(gl::FRAMEBUFFER, state.framebuffer);
        gl::UseProgram(state.program);
        gl::BindBuffer(gl::ARRAY_BUFFER, state.vb);
        gl::BindVertexArray(state.vao);
    }
}

/// Sets up the OpenGL stuff for interleaving the array of 2D textures to send to glass
fn setup_blend(glass: &LookingGlass) -> GlState {
    const VERTEX_SHADER: &'static [u8] = b"
    #version 330
    precision mediump float;

    in vec2 position;
    in vec2 texcoord;

    out vec2 v_texcoord;

    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        v_texcoord = texcoord;
    }
    \0";

    let fragment_shader = format!("
    #version 330
    precision mediump float;

    uniform sampler2DArray sampler;

    in vec2 v_texcoord;

    out vec4 fragColor;

    const int width = {width};
    const int height = {height};
    const int dpi = {dpi};

    const float slope_cal = {slope_cal};
    const float center_cal = {center_cal};
    const float pitch_cal = {pitch_cal};

    const float tilt = height / (width*slope_cal);
    const float pitch = pitch_cal * width / dpi * cos(atan(1.0, slope_cal));
    const float subp = 1.0 / (3*width);

    const float center = center_cal - tilt * pitch;

    void main() {{
      float a = (v_texcoord.x - tilt * v_texcoord.y)*pitch - center;

      fragColor.r = texture(sampler, vec3(v_texcoord, 45.0 * fract(a))).r;
      fragColor.g = texture(sampler, vec3(v_texcoord, 45.0 * fract(a+subp))).g;
      fragColor.b = texture(sampler, vec3(v_texcoord, 45.0 * fract(a+2*subp))).b;
      fragColor.a = 1.0;
    }}\0",
    width = glass.screen_w,
    height = glass.screen_h,
    dpi = glass.dpi,
    slope_cal = glass.slope,
    center_cal = glass.center,
    pitch_cal = glass.pitch,
    );

    let program;
    let mut vb;
    let mut vao;
    unsafe {
        let vs = gl::CreateShader(gl::VERTEX_SHADER);
        gl::ShaderSource(vs, 1, [VERTEX_SHADER.as_ptr() as *const _].as_ptr(), ptr::null());
        gl::CompileShader(vs);

        let fs = gl::CreateShader(gl::FRAGMENT_SHADER);
        gl::ShaderSource(fs, 1, [fragment_shader.as_ptr() as *const _].as_ptr(), ptr::null());
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
        framebuffer: 0, // Render the blends to the screen
        vb,
        program,
        vao,
    }
}

/// Sets up the OpenGL stuff for rendering the 3D scene to 2D views
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
