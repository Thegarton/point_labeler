#include "ImageViewer.h"

#include <QtCore/QPointF>
#include <QtGui/QColor>
#include <QtGui/QPen>
#include <cmath>

ImageViewer::ImageViewer(QWidget* parent, Qt::WindowFlags f) : QWidget(parent, f) {}

void ImageViewer::setImage(const std::string& filename, const std::vector<Eigen::Vector2f>* projected_points) {
  currentImage_ = QPixmap(QString::fromStdString(filename));
  if (projected_points != nullptr) {
    projectedPoints_ = *projected_points;
  } else {
    projectedPoints_.clear();
  }
  update();
}

void ImageViewer::paintEvent(QPaintEvent* event) {
  QPainter painter(this);

  painter.drawPixmap(0, 0, width(), height(), currentImage_);
  if (currentImage_.isNull() || projectedPoints_.empty()) return;

  const float scale_x = float(width()) / float(currentImage_.width());
  const float scale_y = float(height()) / float(currentImage_.height());
  painter.setPen(QPen(QColor(255, 230, 0, 210), 2));
  for (const Eigen::Vector2f& point : projectedPoints_) {
    if (!std::isfinite(point.x()) || !std::isfinite(point.y())) continue;
    if (point.x() < 0.0f || point.y() < 0.0f || point.x() >= currentImage_.width() ||
        point.y() >= currentImage_.height())
      continue;
    painter.drawPoint(QPointF(point.x() * scale_x, point.y() * scale_y));
  }
}

void ImageViewer::resizeEvent(QResizeEvent* event) { update(); }
