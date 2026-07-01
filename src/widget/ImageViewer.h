#ifndef SRC_WIDGET_IMAGEVIEWER_H_
#define SRC_WIDGET_IMAGEVIEWER_H_

#include <QtGui/QMouseEvent>
#include <QtGui/QPainter>
#include <QtGui/QPixmap>
#include <QtWidgets/QWidget>
#include <eigen3/Eigen/Dense>
#include <string>
#include <vector>

/** \brief show an image.
 *  \author behley
 **/
class ImageViewer : public QWidget {
  Q_OBJECT
 public:
  ImageViewer(QWidget* parent = 0, Qt::WindowFlags f = 0);

  void setImage(const std::string& filename, const std::vector<Eigen::Vector2f>* projected_points = nullptr);

 protected:
  void paintEvent(QPaintEvent* event);
  void resizeEvent(QResizeEvent* event);

  std::string imageFilename;
  QPixmap currentImage_;
  std::vector<Eigen::Vector2f> projectedPoints_;
};

#endif /* SRC_WIDGET_IMAGEVIEWER_H_ */
