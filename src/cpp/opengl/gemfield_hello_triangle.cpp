#include<X11/Xlib.h>
#include<X11/Xutil.h>
#include<memory.h>
#include<EGL/egl.h>
#include<GLES2/gl2.h>
#include<iostream>
#include<vector>
#include<unistd.h>

#include <string>
#include <fstream>
#include <streambuf>
//g++ gemfield_hello_triangle.cpp -o gemfield -lEGL -lGLESv2 -lX11
class GemfieldWin{
public:
    GemfieldWin(int window_width = 800, int window_height = 500){
        window_width_ = window_width;
        window_height_ = window_height;
    }

    bool openDisplay(){
        x_display_ = XOpenDisplay(NULL);
        if ( x_display_ == NULL ){
            return false;
        }
        //Gemfield: find the screen
        Screen* screen = DefaultScreenOfDisplay(x_display_);
        //Gemfield: find the screen ID of the screen
	    int screen_id = DefaultScreen(x_display_);
        Window window = XCreateSimpleWindow(x_display_, RootWindowOfScreen(screen), 0, 0, window_width_, window_height_, 1, BlackPixel(x_display_, screen_id), WhitePixel(x_display_, screen_id));
        Atom s_wmDeleteMessage = XInternAtom(x_display_, "WM_DELETE_WINDOW", False);
	    XSetWMProtocols(x_display_, window, &s_wmDeleteMessage, 1);
        XSetWindowAttributes  xattr;
	    xattr.override_redirect = false;
	    XChangeWindowAttributes ( x_display_, window, CWOverrideRedirect, &xattr );

        XWMHints hints;
        hints.input = true;
        hints.flags = InputHint;
        XSetWMHints(x_display_, window, &hints);

        // make the window visible on the screen
        XMapWindow (x_display_, window);
        XStoreName (x_display_, window, "SYSZUXshader- esc exit");

        // get identifiers for the provided atom name strings
        Atom wm_state;
        wm_state = XInternAtom (x_display_, "_NET_WM_STATE", false);
        XEvent xev;
        memset ( &xev, 0, sizeof(xev) );
        xev.type                 = ClientMessage;
        xev.xclient.window       = window;
        xev.xclient.message_type = wm_state;
        xev.xclient.format       = 32;
        xev.xclient.data.l[0]    = 1;
        xev.xclient.data.l[1]    = false;
        XSendEvent (x_display_,RootWindowOfScreen(screen),false,SubstructureNotifyMask,&xev );
        XSelectInput(x_display_, window, ExposureMask|KeyPressMask|StructureNotifyMask|ButtonPressMask);

        EGLNativeWindowType eglNativeWindow = (EGLNativeWindowType) window;
        
        const EGLint configAttribs[] = { EGL_RENDERABLE_TYPE,EGL_WINDOW_BIT,EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_DEPTH_SIZE,24,EGL_NONE };
        const EGLint contextAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
        egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (egl_display_ == EGL_NO_DISPLAY)
            return false;

        EGLint major, minor;
        if (!eglInitialize(egl_display_, &major, &minor))
            return false;

        EGLConfig config;
        EGLint numConfigs;
        if (!eglChooseConfig(egl_display_, configAttribs, &config, 1, &numConfigs))
            return false;
        egl_surface_ = eglCreateWindowSurface(egl_display_, config, eglNativeWindow, NULL);
        if (egl_surface_ == EGL_NO_SURFACE)
            return false;
        EGLContext eglContext = eglCreateContext(egl_display_, config, EGL_NO_CONTEXT, contextAttribs);
        if (eglContext == EGL_NO_CONTEXT)
            return false;
        if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, eglContext))
            return false;
        return true;
    }

    std::string getShaderByte(const char* path){
        std::ifstream t(path);
        std::string str((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());
        return str;
    }

    void show(){
        // Create a Vertex Buffer Object and copy the vertex data to it
        GLuint vbo;
        glGenBuffers(1, &vbo);

        GLfloat vertices[] = {0.0f, 0.5f, 0.5f, -0.5f, -0.5f, -0.5f};

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        // Create and compile the vertex shader
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        //move 
        std::string tmp_vertex_str = getShaderByte("vertex.glsl");
        std::string tmp_frag_str = getShaderByte("fragment.glsl");

        const char* vertex_shader = tmp_vertex_str.c_str();
        const char* frag_shader = tmp_frag_str.c_str();

        glShaderSource(vertexShader, 1, &vertex_shader, NULL);
        glCompileShader(vertexShader);

        // Create and compile the fragment shader
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &frag_shader, NULL);
        glCompileShader(fragmentShader);

        // Link the vertex and fragment shader into a shader program
        GLuint shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        // glBindFragDataLocation(shaderProgram, 0, "outColor");
        glLinkProgram(shaderProgram);
        glUseProgram(shaderProgram);

        // Specify the layout of the vertex data
        GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
        glEnableVertexAttribArray(posAttrib);
        glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0);

        while(true){
            // Clear the screen to black
            //glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            //glClear(GL_COLOR_BUFFER_BIT);

            // Draw a triangle from the 3 vertices
            glDrawArrays(GL_TRIANGLES, 0, 3);
            eglSwapBuffers(egl_display_,egl_surface_);
            usleep(10000000);
        }
    }
private:
    Display *x_display_;
    int window_width_{800};
    int window_height_{500};
    Window screen_;
    EGLDisplay egl_display_;
    EGLSurface egl_surface_;
};

int main()
{
    GemfieldWin gwin(1400,900);
    gwin.openDisplay();
    gwin.show();
    return 0;
}
