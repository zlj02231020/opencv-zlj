// ============================================================
// 第二题：硬币检测与面值统计
//
// 思路：
//   1. 硬币是圆形的、比背景暗，直接用 HoughCircles 在灰度图上找圆。
//   2. 题目要求"自动判断尺寸分类，不能硬编码位置或数量"。所以半径
//      不用写死的阈值，而是把所有检出的半径丢进 k-means 聚成 3 类，
//      再按类中心的半径大小排序 —— 半径最大的是 1 元，中间是 5 角，
//      最小是 1 角。这样换一张图（硬币大小不同）也能自动适配。
//   3. 按 1 元=1.0、5 角=0.5、1 角=0.1 汇总总金额。
// ============================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    string inPath = (argc > 1) ? argv[1] : "images/q02_test.png";
    Mat img = imread(inPath);
    if (img.empty()) {
        cerr << "无法读取图像: " << inPath << endl;
        return 1;
    }

    Mat gray0;
    cvtColor(img, gray0, COLOR_BGR2GRAY);
    Mat gray;
    // 轻度降噪，避免纹理干扰霍夫变换
    GaussianBlur(gray0, gray, Size(5, 5), 1.5);

    // ---------- 1. 霍夫圆检测 ----------
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT,
                 1,          // dp：累加器分辨率与原图一致
                 25,         // minDist：两枚硬币圆心的最小距离
                 100,        // param1：Canny 高阈值
                 30,         // param2：累加器阈值，越小检出越多
                 15,         // minRadius
                 70);        // maxRadius

    if (circles.empty()) {
        cerr << "没有检测到硬币" << endl;
        return 2;
    }

    // ---------- 1.5 半径精修 ----------
    // 霍夫给出的半径是累加器的峰值位置，普遍偏小 2~3 像素（实测 1元
    // 币给 48~52，真值 50）。硬币是一块比背景暗的实心圆，所以离边界
    // 最远的点的距离变换值就是半径：在圆心附近取距离变换的峰值，既
    // 能把半径订正准确，也顺手把圆心挪到正中间。
    // 背景亮度取四条边的中位数（四边一定是背景），比 Otsu 稳：这枚
    // 硬币币面上有一半像素是深色花纹，Otsu 会把阈值切到花纹与币面之间，
    // 硬币内部被打出一堆洞，距离变换就只剩 2 像素了。
    vector<uchar> borderPix;
    for (int x = 0; x < gray0.cols; ++x) {
        borderPix.push_back(gray0.at<uchar>(0, x));
        borderPix.push_back(gray0.at<uchar>(gray0.rows - 1, x));
    }
    for (int y = 0; y < gray0.rows; ++y) {
        borderPix.push_back(gray0.at<uchar>(y, 0));
        borderPix.push_back(gray0.at<uchar>(y, gray0.cols - 1));
    }
    nth_element(borderPix.begin(), borderPix.begin() + borderPix.size() / 2, borderPix.end());
    int bgLevel = (int)borderPix[borderPix.size() / 2];
    int cut = bgLevel - 20;

    Mat fg;
    threshold(gray0, fg, cut, 255, THRESH_BINARY_INV);
    Mat dist;
    distanceTransform(fg, dist, DIST_L2, 5);

    for (size_t i = 0; i < circles.size(); ++i) {
        int cx = cvRound(circles[i][0]), cy = cvRound(circles[i][1]);
        int half = 4;
        int x0 = max(0, cx - half), y0 = max(0, cy - half);
        int x1 = min(dist.cols, cx + half + 1), y1 = min(dist.rows, cy + half + 1);
        Mat roi = dist(Rect(x0, y0, x1 - x0, y1 - y0));
        double dmn, dmx;
        Point loc;
        minMaxLoc(roi, &dmn, &dmx, NULL, &loc);
        if (dmx > 5) {
            circles[i][0] = (float)(x0 + loc.x);
            circles[i][1] = (float)(y0 + loc.y);
            circles[i][2] = (float)dmx;
        }
    }

    // ---------- 2. 用 k-means 把半径自动分成 3 档 ----------
    Mat samples((int)circles.size(), 1, CV_32F);
    for (size_t i = 0; i < circles.size(); ++i)
        samples.at<float>((int)i, 0) = circles[i][2];

    Mat labels, centers;
    kmeans(samples, 3, labels,
           TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1),
           10, KMEANS_PP_CENTERS, centers);

    // 按类中心半径从小到大排序，建立 类编号 -> 面值 的映射
    vector<pair<float, int>> order;   // (中心半径, 原类号)
    for (int i = 0; i < 3; ++i)
        order.push_back({centers.at<float>(i, 0), i});
    sort(order.begin(), order.end());

    const char* names[3] = {"1角", "5角", "1元"};
    const double values[3] = {0.1, 0.5, 1.0};
    const Scalar colors[3] = {Scalar(255, 0, 0),     // 1角 - 蓝
                              Scalar(0, 200, 0),     // 5角 - 绿
                              Scalar(0, 0, 255)};    // 1元 - 红
    int classOf[3];
    for (int k = 0; k < 3; ++k) classOf[order[k].second] = k;  // 原类号 -> 档位

    int count[3] = {0, 0, 0};
    double total = 0.0;
    Mat out = img.clone();

    for (size_t i = 0; i < circles.size(); ++i) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        int k = classOf[labels.at<int>((int)i, 0)];

        ++count[k];
        total += values[k];

        circle(out, center, radius, colors[k], 2);
        circle(out, center, 2, colors[k], -1);
        putText(out, names[k], center + Point(-18, radius + 18),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 3, LINE_AA);
        putText(out, names[k], center + Point(-18, radius + 18),
                FONT_HERSHEY_SIMPLEX, 0.5, colors[k], 1, LINE_AA);
    }

    // ---------- 3. 输出 ----------
    cout << "检出硬币总数: " << circles.size() << endl;
    for (int k = 2; k >= 0; --k)
        cout << names[k] << ": " << count[k] << " 个（该档平均半径 "
             << (int)order[k].first << " px）" << endl;
    cout << "总金额: " << fixed << setprecision(1) << total << " 元" << endl;

    char head[128];
    snprintf(head, sizeof(head), "1yuan:%d  5jiao:%d  1jiao:%d  total:%.1f",
             count[2], count[1], count[0], total);
    putText(out, head, Point(10, 28), FONT_HERSHEY_SIMPLEX, 0.7,
            Scalar(0, 0, 0), 2, LINE_AA);

    string outPath = (argc > 2) ? argv[2] : "q02_result.png";
    imwrite(outPath, out);
    cout << "结果图已保存: " << outPath << endl;
    return 0;
}
