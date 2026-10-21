#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/features/shot.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: pcl_descriptor_export input.pcd output.csv radius_m\n";
    return 2;
  }

  const std::string input_path = argv[1];
  const std::string output_path = argv[2];
  const float radius = std::strtof(argv[3], nullptr);
  if (!std::isfinite(radius) || radius <= 0.0F) {
    std::cerr << "Feature radius must be a positive finite value in meters\n";
    return 2;
  }

  using Point = pcl::PointXYZ;
  using Cloud = pcl::PointCloud<Point>;
  Cloud::Ptr cloud(new Cloud);
  if (pcl::io::loadPCDFile<Point>(input_path, *cloud) < 0) {
    std::cerr << "Could not read input PCD: " << input_path << '\n';
    return 1;
  }
  if (cloud->size() < 12) {
    std::cerr << "At least 12 points are required to estimate descriptors\n";
    return 1;
  }

  pcl::search::KdTree<Point>::Ptr tree(new pcl::search::KdTree<Point>);
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<Point, pcl::Normal> normal_estimator;
  normal_estimator.setInputCloud(cloud);
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setKSearch(std::min<int>(10, static_cast<int>(cloud->size()) - 1));
  normal_estimator.compute(*normals);

  pcl::PFHEstimation<Point, pcl::Normal, pcl::PFHSignature125> pfh_estimator;
  pfh_estimator.setInputCloud(cloud);
  pfh_estimator.setInputNormals(normals);
  pfh_estimator.setSearchMethod(tree);
  pfh_estimator.setRadiusSearch(radius);
  pcl::PointCloud<pcl::PFHSignature125> pfh;
  pfh_estimator.compute(pfh);

  pcl::SHOTEstimation<Point, pcl::Normal, pcl::SHOT352> shot_estimator;
  shot_estimator.setInputCloud(cloud);
  shot_estimator.setInputNormals(normals);
  shot_estimator.setSearchMethod(tree);
  shot_estimator.setRadiusSearch(radius);
  pcl::PointCloud<pcl::SHOT352> shot;
  shot_estimator.compute(shot);

  if (pfh.size() != cloud->size() || shot.size() != cloud->size()) {
    std::cerr << "PCL returned an incomplete PFH or SHOT descriptor cloud\n";
    return 1;
  }

  std::ofstream output(output_path);
  if (!output) {
    std::cerr << "Could not create descriptor CSV: " << output_path << '\n';
    return 1;
  }
  output << "point_index";
  for (int bin = 0; bin < 125; ++bin) {
    output << ",pfh_" << bin;
  }
  for (int bin = 0; bin < 352; ++bin) {
    output << ",shot_" << bin;
  }
  output << '\n';
  output.precision(9);
  for (std::size_t point_index = 0; point_index < cloud->size(); ++point_index) {
    output << point_index;
    for (float value : pfh[point_index].histogram) {
      output << ',' << value;
    }
    for (float value : shot[point_index].descriptor) {
      output << ',' << value;
    }
    output << '\n';
  }
  if (!output) {
    std::cerr << "Failed while writing descriptor CSV: " << output_path << '\n';
    return 1;
  }
  return 0;
}
