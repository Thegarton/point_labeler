#include <stdint.h>
#include <widget/KittiReader.h>

#include <QtCore/QDir>
#include <QtCore/QFileInfo>
#include <QtGui/QColor>
#include <QtGui/QImage>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <sstream>

#include "rv/string_utils.h"

namespace {

std::string lowerTrimmed(std::string value) {
  value = rv::trim(value);
  if (value.size() >= 3 && static_cast<unsigned char>(value[0]) == 0xEF &&
      static_cast<unsigned char>(value[1]) == 0xBB && static_cast<unsigned char>(value[2]) == 0xBF) {
    value = value.substr(3);
  }
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
  return value;
}

std::vector<std::string> splitTableRow(const std::string& line) {
  if (line.find(',') != std::string::npos) {
    std::vector<std::string> tokens = rv::split(line, ",", false);
    for (std::string& token : tokens) token = rv::trim(token);
    return tokens;
  }

  std::vector<std::string> tokens;
  std::stringstream ss(line);
  std::string token;
  while (ss >> token) tokens.push_back(token);
  return tokens;
}

std::map<std::string, uint32_t> parseHeaderColumns(const std::vector<std::string>& header) {
  std::map<std::string, uint32_t> columns;
  for (uint32_t i = 0; i < header.size(); ++i) {
    std::string name = lowerTrimmed(header[i]);
    if (!name.empty() && columns.find(name) == columns.end()) columns[name] = i;
  }
  return columns;
}

uint32_t requireColumn(const std::map<std::string, uint32_t>& columns, const std::string& name,
                       const std::string& filename) {
  auto it = columns.find(name);
  if (it == columns.end()) {
    std::stringstream ss;
    ss << "CSV point cloud " << filename << " is missing required column '" << name << "'.";
    throw std::runtime_error(ss.str());
  }
  return it->second;
}

bool optionalColumn(const std::map<std::string, uint32_t>& columns, const std::string& name, uint32_t& index) {
  auto it = columns.find(name);
  if (it == columns.end()) return false;
  index = it->second;
  return true;
}

float parseCsvFloat(const std::vector<std::string>& tokens, uint32_t column, const std::string& column_name,
                    const std::string& filename, uint32_t line_number) {
  if (column >= tokens.size()) {
    std::stringstream ss;
    ss << "CSV point cloud " << filename << " line " << line_number << " has no value for column '" << column_name
       << "'.";
    throw std::runtime_error(ss.str());
  }

  std::string value = rv::trim(tokens[column]);
  const char* begin = value.c_str();
  char* end = nullptr;
  errno = 0;
  float parsed = std::strtof(begin, &end);
  if (begin == end || errno == ERANGE || !std::isfinite(parsed) || !rv::trim(std::string(end)).empty()) {
    std::stringstream ss;
    ss << "CSV point cloud " << filename << " line " << line_number << " has invalid float in column '"
       << column_name << "': " << value;
    throw std::runtime_error(ss.str());
  }
  return parsed;
}

bool readHeaderLine(std::ifstream& in, std::string& header_line, uint32_t& line_number) {
  while (std::getline(in, header_line)) {
    ++line_number;
    header_line = rv::trim(header_line);
    if (!header_line.empty()) return true;
  }
  return false;
}

}  // namespace

void KittiReader::initialize(const QString& directory) {
  velodyne_filenames_.clear();
  label_filenames_.clear();
  image_filenames_.clear();
  rgb_filenames_.clear();
  projection_image_filenames_.clear();
  poses_.clear();

  pointsCache_.clear();
  labelCache_.clear();
  tiles_.clear();
  maxInstanceIds_.clear();
  calib_.clear();

  base_dir_ = QDir(directory);
  const bool metadataFallback = allowVelodyneOnly_ || readPointsFromCsv_;
  QDir velodyne_dir(readPointsFromCsv_ ? base_dir_.filePath("csv") : base_dir_.filePath("velodyne"));
  if (readPointsFromCsv_ && !velodyne_dir.exists() && base_dir_.exists("CSV")) {
    velodyne_dir = QDir(base_dir_.filePath("CSV"));
  }
  if (!velodyne_dir.exists() && allowVelodyneOnly_ && !readPointsFromCsv_) {
    velodyne_dir = base_dir_;
    std::cout << "-- Missing velodyne/ directory, reading .bin files directly from selected directory." << std::endl;
  }
  QStringList filters;
  if (readPointsFromCsv_) {
    filters << "*.csv"
            << "*.tsv"
            << "*.txt";
  } else {
    filters << "*.bin";
  }
  QStringList entries = velodyne_dir.entryList(filters, QDir::Files, QDir::Name);
  for (int32_t i = 0; i < entries.size(); ++i) {
    velodyne_filenames_.push_back(velodyne_dir.filePath(entries.at(i)).toStdString());
  }

  if (velodyne_filenames_.empty()) {
    std::stringstream ss;
    ss << "No " << (readPointsFromCsv_ ? "CSV" : ".bin") << " point clouds found in "
       << velodyne_dir.path().toStdString();
    throw std::runtime_error(ss.str());
  }

  if (!base_dir_.exists("calib.txt") && !metadataFallback)
    throw std::runtime_error("Missing calibration file: " + base_dir_.filePath("calib.txt").toStdString());

  if (base_dir_.exists("calib.txt")) {
    calib_.initialize(base_dir_.filePath("calib.txt").toStdString());
  } else {
    std::cout << "-- Missing calib.txt, using identity calibration for raw point cloud view mode." << std::endl;
  }

  if (base_dir_.exists("poses.txt")) {
    readPoses(base_dir_.filePath("poses.txt").toStdString(), poses_);
  } else if (metadataFallback) {
    poses_.assign(velodyne_filenames_.size(), Eigen::Matrix4f::Identity());
    std::cout << "-- Missing poses.txt, using identity poses for " << poses_.size()
              << " scans in raw point cloud view mode." << std::endl;
  } else {
    throw std::runtime_error("Missing pose file: " + base_dir_.filePath("poses.txt").toStdString());
  }

  if (poses_.size() != velodyne_filenames_.size()) {
    std::stringstream ss;
    ss << "Number of poses (" << poses_.size() << ") does not match number of point clouds ("
       << velodyne_filenames_.size() << ").";
    throw std::runtime_error(ss.str());
  }

  // create label dir, etc.
  QDir labels_dir(base_dir_.filePath("labels"));

  // find corresponding label files.
  if (!labels_dir.exists()) base_dir_.mkdir("labels");

  for (uint32_t i = 0; i < velodyne_filenames_.size(); ++i) {
    QString filename = QFileInfo(QString::fromStdString(velodyne_filenames_[i])).baseName() + ".label";
    if (!labels_dir.exists(filename)) {
      uint32_t num_points = 0;
      if (readPointsFromCsv_) {
        num_points = countCsvPoints(velodyne_filenames_[i]);
      } else {
        std::ifstream in(velodyne_filenames_[i].c_str());
        in.seekg(0, std::ios::end);
        num_points = in.tellg() / (4 * sizeof(float));
        in.close();
      }

      std::ofstream out(labels_dir.filePath(filename).toStdString().c_str());

      std::vector<uint32_t> labels(num_points, 0);
      out.write(reinterpret_cast<const char*>(labels.data()), num_points * sizeof(uint32_t));

      out.close();
    }

    label_filenames_.push_back(labels_dir.filePath(filename).toStdString());
  }

  std::string missing_img = QDir::currentPath().toStdString() + "/../assets/missing.png";
  QDir image_dir(base_dir_.filePath("image_2"));
  if (!image_dir.exists() && base_dir_.exists("Image_2")) image_dir = QDir(base_dir_.filePath("Image_2"));
  QDir rgb_dir(base_dir_.filePath("point_rgb"));
  for (uint32_t i = 0; i < velodyne_filenames_.size(); ++i) {
    QString filename_base = QFileInfo(QString::fromStdString(velodyne_filenames_[i])).baseName();
    QString filename_jpg = filename_base + ".jpg";
    QString filename_jpeg = filename_base + ".jpeg";
    QString filename_png = filename_base + ".png";
    std::string projection_image;
    if (image_dir.exists(filename_jpg)) {
      projection_image = image_dir.filePath(filename_jpg).toStdString();
      image_filenames_.push_back(projection_image);
    } else if (image_dir.exists(filename_jpeg)) {
      projection_image = image_dir.filePath(filename_jpeg).toStdString();
      image_filenames_.push_back(projection_image);
    } else if (image_dir.exists(filename_png)) {
      projection_image = image_dir.filePath(filename_png).toStdString();
      image_filenames_.push_back(projection_image);
    } else {
      image_filenames_.push_back(missing_img);
    }
    projection_image_filenames_.push_back(projection_image);
    rgb_filenames_.push_back(rgb_dir.filePath(filename_base + ".rgb").toStdString());
  }

  // assumes that (0,0,0) is always the start.
  Eigen::Vector2f min = Eigen::Vector2f::Zero();
  Eigen::Vector2f max = Eigen::Vector2f::Zero();

  for (uint32_t i = 0; i < poses_.size(); ++i) {
    Eigen::Vector4f t = poses_[i].col(3);

    min.x() = std::min(t.x() - maxDistance_, min.x());
    min.y() = std::min(t.y() - maxDistance_, min.y());
    max.x() = std::max(t.x() + maxDistance_, max.x());
    max.y() = std::max(t.y() + maxDistance_, max.y());
  }

  //  std::cout << "tileSize = " << tileSize_ << std::endl;
  //  std::cout << "min = " << min << ", max = " << max << std::endl;

  offset_.x() = std::ceil((std::abs(min.x()) - 0.5 * tileSize_) / tileSize_) * tileSize_ + 0.5 * tileSize_;
  offset_.y() = std::ceil((std::abs(min.y()) - 0.5 * tileSize_) / tileSize_) * tileSize_ + 0.5 * tileSize_;

  //  std::cout << "offset = " << offset_ << std::endl;

  numTiles_.x() = std::ceil((std::abs(min.x()) - 0.5 * tileSize_) / tileSize_) +
                  std::ceil((max.x() - 0.5 * tileSize_) / tileSize_) + 1;
  numTiles_.y() = std::ceil((std::abs(min.y()) - 0.5 * tileSize_) / tileSize_) +
                  std::ceil((max.y() - 0.5 * tileSize_) / tileSize_) + 1;

  //  std::cout << "numTiles = " << numTiles_ << std::endl;

  tiles_.clear();
  tiles_.resize(numTiles_.x() * numTiles_.y());

  Eigen::Vector2f idxRadius(maxDistance_ / tileSize_, maxDistance_ / tileSize_);

  for (uint32_t i = 0; i < uint32_t(numTiles_.x()); ++i) {
    for (uint32_t j = 0; j < uint32_t(numTiles_.y()); ++j) {
      auto& tile = tiles_[tileIdxToOffset(i, j)];

      tile.i = i;
      tile.j = j;
      tile.x = i * tileSize_ - offset_.x() + 0.5 * tileSize_;
      tile.y = j * tileSize_ - offset_.y() + 0.5 * tileSize_;
      tile.size = tileSize_;
    }
  }

  trajectory_.clear();

  Eigen::Vector2f e(0.5 * tileSize_, 0.5 * tileSize_);
  for (uint32_t i = 0; i < poses_.size(); ++i) {
    Eigen::Vector2f t = poses_[i].col(3).head(2);
    Eigen::Vector2f idx((t.x() + offset_.x()) / tileSize_, (t.y() + offset_.y()) / tileSize_);

    trajectory_.push_back(Eigen::Vector2f((t.x() + offset_.x()) / tileSize_, (t.y() + offset_.y()) / tileSize_));

    //    tiles_[tileIdxToOffset(uint32_t(idx.x()), uint32_t(idx.y()))].indexes.push_back(i);
    //    uint32_t u_min = std::max(int32_t(idx.x() - idxRadius.x()), 0);
    //    uint32_t u_max = std::min(int32_t(std::ceil(idx.x() + idxRadius.x())), numTiles_.x());
    //    uint32_t v_min = std::max(int32_t(idx.y() - idxRadius.y()), 0);
    //    uint32_t v_max = std::min(int32_t(std::ceil(idx.y() + idxRadius.y())), numTiles_.y());

    // FIXME: workaround check all tiles.
    for (uint32_t u = 0; u < uint32_t(numTiles_.x()); ++u) {
      for (uint32_t v = 0; v < uint32_t(numTiles_.y()); ++v) {
        auto& tile = tiles_[tileIdxToOffset(u, v)];
        Eigen::Vector2f q = t - Eigen::Vector2f(tile.x, tile.y);
        q[0] = std::abs(q[0]);
        q[1] = std::abs(q[1]);

        // check for exact overlap (see Behley et al., ICRA, 2015)
        if (std::max(q[0], q[1]) > e[0] + maxDistance_) continue;  // definitely outside.
        if (std::min(q[0], q[1]) < e[0] || (q - e).norm() < maxDistance_) {
          tile.indexes.push_back(i);
        }
      }
    }
  }

  // sanity check:

  for (auto& t : tiles_) {
    std::sort(t.indexes.begin(), t.indexes.end());
    //    std::cout << "Tile has " << t.indexes.size() << " tiles associated." << std::endl;
    for (uint32_t i = 1; i < t.indexes.size(); ++i) {
      if (t.indexes[i - 1] == t.indexes[i]) {
        std::cout << "found duplicate!" << std::endl;
      }
    }
  }

  uint32_t tileCount = 0;
  for (uint32_t i = 0; i < uint32_t(numTiles_.x()); ++i) {
    for (uint32_t j = 0; j < uint32_t(numTiles_.y()); ++j) {
      auto& tile = tiles_[tileIdxToOffset(i, j)];

      std::sort(tile.indexes.begin(), tile.indexes.end());
      if (tile.indexes.size() > 0) tileCount += 1;
    }
  }

  std::cout << "#tiles  = " << tileCount << std::endl;

  // meta information for faster loading.
  if (base_dir_.exists("instances.txt")) {
    std::cout << "Reading instances.txt..." << std::flush;

    std::ifstream in(base_dir_.filePath("instances.txt").toStdString());

    while (in.good()) {
      std::string line;
      std::getline(in, line);
      if (line.size() == 0) break;

      std::vector<std::string> tokens = rv::split(line, ":");
      if (tokens.size() != 2) {
        throw std::runtime_error("Invalid instance meta information found!");
      }

      uint32_t label = boost::lexical_cast<uint32_t>(tokens[0]);
      uint32_t maxInstanceId = boost::lexical_cast<uint32_t>(tokens[1]);
      maxInstanceIds_[label] = maxInstanceId;

      in.peek();
    }

    in.close();
    std::cout << "finished." << std::endl;
  } else {
    std::cout << "Generating intances.txt" << std::flush;
    // get the counts from the label files.
    for (const std::string& filename : label_filenames_) {
      std::vector<uint32_t> labels;
      readLabels(filename, labels);

      for (uint32_t instance_label : labels) {
        uint32_t instanceId = (instance_label >> 16) & uint32_t(0xFFFF);
        uint32_t label = instance_label & uint32_t(0xFFFF);
        if (maxInstanceIds_.find(label) == maxInstanceIds_.end())
          maxInstanceIds_[label] = instanceId;
        else
          maxInstanceIds_[label] = std::max(instanceId, maxInstanceIds_[label]);
      }
    }

    // directly update meta information:
    updateMetaInformation(maxInstanceIds_);
  }
}

void KittiReader::updateMetaInformation(const std::map<uint32_t, uint32_t>& maxInstanceIds) {
  std::ofstream out(base_dir_.filePath("instances.txt").toStdString().c_str());
  for (auto it = maxInstanceIds.begin(); it != maxInstanceIds.end(); ++it) {
    out << it->first << ":" << it->second << std::endl;
  }
  out.close();
}

void KittiReader::retrieve(const Eigen::Vector3f& position, std::vector<uint32_t>& indexes,
                           std::vector<PointcloudPtr>& points, std::vector<LabelsPtr>& labels,
                           std::vector<std::string>& images) {
  Eigen::Vector2f idx((position.x() + offset_.x()) / tileSize_, (position.y() + offset_.y()) / tileSize_);

  retrieve(idx.x(), idx.y(), indexes, points, labels, images);
}

void KittiReader::retrieve(uint32_t i, uint32_t j, std::vector<uint32_t>& indexes, std::vector<PointcloudPtr>& points,
                           std::vector<LabelsPtr>& labels, std::vector<std::string>& images) {
  indexes.clear();
  points.clear();
  labels.clear();
  images.clear();

  std::vector<int32_t> indexesBefore;
  for (auto it = pointsCache_.begin(); it != pointsCache_.end(); ++it) indexesBefore.push_back(it->first);
  std::vector<int32_t> indexesAfter;

  uint32_t scansRead = 0;

  indexes = tiles_[tileIdxToOffset(i, j)].indexes;
  std::sort(indexes.begin(), indexes.end());
  for (uint32_t t : indexes) {
    indexesAfter.push_back(t);
    if (pointsCache_.find(t) == pointsCache_.end()) {
      scansRead += 1;

      points.push_back(std::shared_ptr<Laserscan>(new Laserscan));
      if (readPointsFromCsv_) {
        readCsvPoints(velodyne_filenames_[t], projection_image_filenames_[t], *points.back());
      } else {
        readPoints(velodyne_filenames_[t], rgb_filenames_[t], *points.back());
      }
      pointsCache_[t] = points.back();
      points.back()->pose = poses_[t];

      labels.push_back(std::shared_ptr<std::vector<uint32_t>>(new std::vector<uint32_t>()));
      readLabels(label_filenames_[t], *labels.back());
      labelCache_[t] = labels.back();

      if (points.back()->size() != labels.back()->size()) {
        std::cout << "Filename: " << velodyne_filenames_[t] << std::endl;
        std::cout << "Filename: " << label_filenames_[t] << std::endl;
        std::cout << "num. points = " << points.back()->size() << " vs. num. labels = " << labels.back()->size()
                  << std::endl;
        throw std::runtime_error("Inconsistent number of labels.");
      }

    } else {
      points.push_back(pointsCache_[t]);
      labels.push_back(labelCache_[t]);
    }

    images.push_back(image_filenames_[t]);
  }

  std::cout << scansRead << " point clouds read." << std::endl;

  // FIXME: keep more scans in cache. not only remove unloaded scans.

  std::sort(indexesBefore.begin(), indexesBefore.end());
  std::sort(indexesAfter.begin(), indexesAfter.end());

  std::vector<int32_t> needsDelete(indexesBefore.size());
  std::vector<int32_t>::iterator end = std::set_difference(
      indexesBefore.begin(), indexesBefore.end(), indexesAfter.begin(), indexesAfter.end(), needsDelete.begin());

  for (auto it = needsDelete.begin(); it != end; ++it) {
    pointsCache_.erase(*it);
    labelCache_.erase(*it);
  }
}

const KittiReader::Tile& KittiReader::getTile(const Eigen::Vector3f& position) const {
  Eigen::Vector2f idx((position.x() + offset_.x()) / tileSize_, (position.y() + offset_.y()) / tileSize_);
  return tiles_[tileIdxToOffset(idx.x(), idx.y())];
}
const KittiReader::Tile& KittiReader::getTile(uint32_t i, uint32_t j) const { return tiles_[tileIdxToOffset(i, j)]; }

void KittiReader::setTileSize(float size) { tileSize_ = size; }

void KittiReader::update(const std::vector<uint32_t>& indexes, std::vector<LabelsPtr>& labels) {

  for (uint32_t i = 0; i < indexes.size(); ++i) {
    if (labels[i]->size() == 0) {
      std::cout << "Warning: 0 labels?" << std::endl;
      continue;
    }

    if (pointsCache_.find(indexes[i]) == pointsCache_.end()) {
      std::cout << "Warning: labels of non cached points?" << std::endl;
      continue;
    }

    if (labels[i]->size() != pointsCache_[indexes[i]]->size()) {
      std::cout << "Warning: inconsistent numbers of labels for given point cloud!" << std::endl;

      continue;
    }

    if (label_filenames_.size() < indexes[i]) {
      std::cout << "Warning: wrong index?" << std::endl;

      continue;
    }

    std::ofstream out(label_filenames_[indexes[i]].c_str());
    out.write((const char*)&(*labels[i])[0], labels[i]->size() * sizeof(uint32_t));
    out.close();
  }
}

void KittiReader::readPoints(const std::string& filename, Laserscan& scan) {
  readPoints(filename, "", scan);
}

void KittiReader::readPoints(const std::string& filename, const std::string& rgb_filename, Laserscan& scan) {
  std::ifstream in(filename.c_str(), std::ios::binary);
  if (!in.is_open()) return;

  scan.clear();

  in.seekg(0, std::ios::end);
  uint32_t num_points = in.tellg() / (4 * sizeof(float));
  in.seekg(0, std::ios::beg);

  std::vector<float> values(4 * num_points);
  in.read((char*)&values[0], 4 * num_points * sizeof(float));

  in.close();
  std::vector<Point3f>& points = scan.points;
  std::vector<float>& remissions = scan.remissions;
  std::vector<Point3f>& colors = scan.colors;

  points.resize(num_points);
  remissions.resize(num_points);
  colors.resize(num_points);

  for (uint32_t i = 0; i < num_points; ++i) {
    points[i].x = values[4 * i];
    points[i].y = values[4 * i + 1];
    points[i].z = values[4 * i + 2];
    remissions[i] = values[4 * i + 3];
    colors[i].x = 0.0f;
    colors[i].y = 0.0f;
    colors[i].z = 0.0f;
  }

  if (rgb_filename.size() > 0) readPointColors(rgb_filename, num_points, colors);
}

uint32_t KittiReader::countCsvPoints(const std::string& filename) const {
  std::ifstream in(filename.c_str());
  if (!in.is_open()) {
    throw std::runtime_error("Unable to open CSV point cloud: " + filename);
  }

  std::string line;
  uint32_t line_number = 0;
  if (!readHeaderLine(in, line, line_number)) {
    throw std::runtime_error("CSV point cloud is empty: " + filename);
  }
  std::map<std::string, uint32_t> columns = parseHeaderColumns(splitTableRow(line));
  requireColumn(columns, "x", filename);
  requireColumn(columns, "y", filename);
  requireColumn(columns, "z", filename);
  requireColumn(columns, "intensity", filename);

  uint32_t num_points = 0;
  while (std::getline(in, line)) {
    if (!rv::trim(line).empty()) ++num_points;
  }
  return num_points;
}

void KittiReader::readCsvPoints(const std::string& filename, const std::string& image_filename, Laserscan& scan) {
  std::ifstream in(filename.c_str());
  if (!in.is_open()) {
    throw std::runtime_error("Unable to open CSV point cloud: " + filename);
  }

  scan.clear();

  std::string line;
  uint32_t line_number = 0;
  if (!readHeaderLine(in, line, line_number)) {
    throw std::runtime_error("CSV point cloud is empty: " + filename);
  }

  std::vector<std::string> header = splitTableRow(line);
  std::map<std::string, uint32_t> columns = parseHeaderColumns(header);
  const uint32_t x_col = requireColumn(columns, "x", filename);
  const uint32_t y_col = requireColumn(columns, "y", filename);
  const uint32_t z_col = requireColumn(columns, "z", filename);
  const uint32_t intensity_col = requireColumn(columns, "intensity", filename);

  uint32_t cxd_col = 0;
  uint32_t cyd_col = 0;
  const bool has_projection = optionalColumn(columns, "cxd", cxd_col) && optionalColumn(columns, "cyd", cyd_col);

  QImage image;
  bool sample_image_colors = false;
  if (has_projection && !image_filename.empty()) {
    image = QImage(QString::fromStdString(image_filename));
    if (!image.isNull()) {
      sample_image_colors = true;
      if (csvProjectionImageWidth_ > 0 && csvProjectionImageHeight_ > 0 &&
          (uint32_t(image.width()) != csvProjectionImageWidth_ ||
           uint32_t(image.height()) != csvProjectionImageHeight_)) {
        std::cout << "-- Warning: image size for " << image_filename << " is " << image.width() << "x"
                  << image.height() << ", while CSV projection settings expect " << csvProjectionImageWidth_ << "x"
                  << csvProjectionImageHeight_ << "." << std::endl;
      }
    }
  }

  std::vector<Point3f>& points = scan.points;
  std::vector<float>& remissions = scan.remissions;
  std::vector<Point3f>& colors = scan.colors;
  std::vector<Eigen::Vector2f>& image_points = scan.image_points;

  while (std::getline(in, line)) {
    ++line_number;
    if (rv::trim(line).empty()) continue;

    std::vector<std::string> tokens = splitTableRow(line);
    Point3f point;
    point.x = parseCsvFloat(tokens, x_col, "x", filename, line_number);
    point.y = parseCsvFloat(tokens, y_col, "y", filename, line_number);
    point.z = parseCsvFloat(tokens, z_col, "z", filename, line_number);
    float intensity = parseCsvFloat(tokens, intensity_col, "intensity", filename, line_number);

    Point3f color;
    if (has_projection) {
      float cxd = parseCsvFloat(tokens, cxd_col, "Cxd", filename, line_number);
      float cyd = parseCsvFloat(tokens, cyd_col, "Cyd", filename, line_number);
      image_points.push_back(Eigen::Vector2f(cxd, cyd));
      if (sample_image_colors) {
        int32_t u = int32_t(std::round(cxd));
        int32_t v = int32_t(std::round(cyd));
        if (u >= 0 && v >= 0 && u < image.width() && v < image.height()) {
          QColor pixel = image.pixelColor(u, v);
          color.x = float(pixel.red()) / 255.0f;
          color.y = float(pixel.green()) / 255.0f;
          color.z = float(pixel.blue()) / 255.0f;
        }
      }
    }

    points.push_back(point);
    remissions.push_back(intensity);
    colors.push_back(color);
  }
}

void KittiReader::readPointColors(const std::string& filename, uint32_t num_points, std::vector<Point3f>& colors) {
  std::ifstream in(filename.c_str(), std::ios::binary);
  if (!in.is_open()) return;

  in.seekg(0, std::ios::end);
  uint32_t num_bytes = in.tellg();
  in.seekg(0, std::ios::beg);

  uint32_t expected_bytes = num_points * 3;
  if (num_bytes != expected_bytes) {
    std::stringstream ss;
    ss << "Invalid RGB file size for " << filename << ": expected " << expected_bytes << " bytes, got "
       << num_bytes;
    throw std::runtime_error(ss.str());
  }

  std::vector<uint8_t> values(num_bytes);
  in.read((char*)&values[0], num_bytes);
  in.close();

  for (uint32_t i = 0; i < num_points; ++i) {
    colors[i].x = float(values[3 * i]) / 255.0f;
    colors[i].y = float(values[3 * i + 1]) / 255.0f;
    colors[i].z = float(values[3 * i + 2]) / 255.0f;
  }
}

void KittiReader::readLabels(const std::string& filename, std::vector<uint32_t>& labels) {
  std::ifstream in(filename.c_str(), std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "Unable to open label file. " << std::endl;
    return;
  }

  labels.clear();

  in.seekg(0, std::ios::end);
  uint32_t num_points = in.tellg() / (sizeof(uint32_t));
  in.seekg(0, std::ios::beg);

  labels.resize(num_points);
  in.read((char*)&labels[0], num_points * sizeof(uint32_t));

  in.close();
}

void KittiReader::readPoses(const std::string& filename, std::vector<Eigen::Matrix4f>& poses) {
  poses = KITTI::Odometry::loadPoses(filename);

  // convert from camera to velodyne coordinate system.
  Eigen::Matrix4f Tr = Eigen::Matrix4f::Identity();
  if (calib_.exists("Tr")) {
    Tr = calib_["Tr"];
  } else {
    std::cout << "-- Calibration matrix Tr is missing, using identity pose calibration." << std::endl;
  }
  Eigen::Matrix4f Tr_inv = Tr.inverse();
  for (uint32_t i = 0; i < poses.size(); ++i) {
    poses[i] = Tr_inv * poses[i] * Tr;
  }
}
