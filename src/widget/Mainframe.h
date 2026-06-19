#ifndef MAINFRAME_H_
#define MAINFRAME_H_

#include <stdint.h>
#include <QtCore/QSignalMapper>
#include <QtCore/QString>
#include <QtGui/QColor>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QProgressBar>
#include <future>
#include "ImageViewer.h"
#include "KittiReader.h"
#include "LabelButton.h"
#include "common.h"
#include "data/geometry.h"
#include "data/label_utils.h"
#include "data/transform.h"
#include "ui_MainFrame.h"
#include "waitingspinnerwidget.h"

/** \brief main widget showing the point cloud and tools to label a point cloud/multiple point clouds. **/
class Mainframe : public QMainWindow {
  Q_OBJECT
 public:
  Mainframe();
  ~Mainframe();

 public slots:
  void open();
  void save();
  void changeRadius(int radius);
  void changeMode(int mode, bool checked);

  void updateFiltering(bool value);
  void labelBtnReleased(QWidget*);

 signals:
  void readerStarted();
  void readerFinshed();

 protected:
  void readAsync(uint32_t i, uint32_t j);

  void updateScans();
  void activateSpinner();

  void forward();
  void backward();

  void setTileIndex(uint32_t i, uint32_t j);

  void setCurrentScanIdx(int32_t idx);

  void generateLabelButtons();
  void clearLabelButtons();
  void reloadLabelDefinitions();
  void addLabelClass();
  void changeSelectedLabelColor();
  void selectLabelById(uint32_t labelId);
  bool appendLabelDefinition(const QString& name, const QColor& color, uint32_t* newLabelId);
  bool updateLabelColor(uint32_t labelId, const QColor& color);
  void closeEvent(QCloseEvent* event);

  void readConfig(const std::string& filename = "settings.cfg");
  void readLabelConfig(const std::string& filename = "settings.cfg");

  void initializeIcons();

  void updateMovingStatus(bool isMoving);

  void updateLabelButtons();

  std::vector<uint32_t> indexes_;
  std::vector<PointcloudPtr> points_;
  std::vector<LabelsPtr> labels_;
  std::vector<std::string> images_;

  std::vector<uint32_t> filteredLabels;
  std::string filename;
  std::string labelFilename_{"labels.xml"};
  bool allowVelodyneOnly_{false};

  void keyPressEvent(QKeyEvent* event);
  void keyReleaseEvent(QKeyEvent* event);

 protected slots:
  void unsavedChanges();

 private:
  Ui::MainWindow ui;
  QSignalMapper* labelButtonMapper{nullptr};
  std::vector<Label> labelDefinitions_;
  std::vector<LabelButton*> labelButtons;
  std::map<LabelButton*, int32_t> labelButtonIdx_;
  std::map<uint32_t, std::string> label_names;
  int32_t selectedLabelButtonIdx_{-1};

  std::map<std::string, std::vector<LabelButton*>> catButtons_;

  bool mChangesSinceLastSave;
  QString lastDirectory;

  Point3f midpoint;

  KittiReader reader_;
  std::future<void> readerFuture_;
  WaitingSpinnerWidget* spinner{nullptr};

  ImageViewer* wImgWidget_;
  QTimer mSaveTimer_;
  QLabel lblLabelingMode_;
  QLabel lblNumPoints_;
  QLabel lblOverwrite_;
  QLabel lblTime_;
  QProgressBar progressLabeled_;
  QWidget* info_;

  struct ScanRange {
    uint32_t start, end;
  };
  std::vector<ScanRange> loopRanges_;
  uint32_t numSelectedInstances_{0};
  QTime mStartLabelTime_;
  QTimer mLabelTimer_;
};

#endif /* MAINFRAME_H_ */
