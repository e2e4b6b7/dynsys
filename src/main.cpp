#include <QApplication>

#include "Window.hpp"

int main(int argc, char *argv[]) {
    QApplication::setAttribute(Qt::AA_UseDesktopOpenGL);
    QApplication app(argc, argv);

    Window w;

    w.resize(w.sizeHint());

    w.showMaximized();

    return app.exec();
}
