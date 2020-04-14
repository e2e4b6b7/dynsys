#pragma once

#include <QGLWidget>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QKeyEvent>
#include <QTime>
#include <QTimer>
#include "Camera.h"
#include "Locus.h"
#include "VideoEncoder.h"

class PointsViewQGLWidget : public QGLWidget {
public:
    explicit PointsViewQGLWidget(QWidget *parent = nullptr);
    ~PointsViewQGLWidget() = default;

    void clearAll();

    void keyPressEvent(QKeyEvent *event) override;
    void keyReleaseEvent(QKeyEvent *event) override;

    void setCurrentTime(const int currentTime_);

    void addNewLocus(QVector<QVector3D> &&points);

    void fixSizes();

protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;

    void mouseMoveEvent(QMouseEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;


private:
    Q_OBJECT

    Locus::LocusController locusController;

    Camera::KeyboardAndMouseController cameraController;

    VideoEncoder::VideoEncoder videoEncoder;

    int currentTime;
};
