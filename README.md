# Looking Glass spinning cube
Uses OpenGL to render a spinning cube in a Looking Glass 3D display

This is a total hack - I'm still learning Rust, haven't used OpenGL in a decade, and am just tinkering around with a Looking Glass.
It would be great if someone more knowledgable with OpenGL or Rust could let me know about any issues with this code - either file an issue here, submit a PR, drop me an email...

To run, you'll probably want cargo installed.  Then, clone and cd in to this repo, do `cargo run`.

The basic idea is that the 3D scene is created once, loaded in to GPU memory, then rendered 45 times in to "layers" of a 2D texture array from slightly different perspectives.
Then, a fullscreen rectangle is drawn using a fragment shader that interleaves those 45 views on to the screen as appropriate for the connected Looking Glass.
