// ============================================================
// 第八题：颜色区域分割与面积计算
//
// 颜色分割的标准做法是在 HSV 里做：
//   - 色调 H 直接对应"是什么颜色"，和明暗无关；饱和度 S 排除灰白，
//     明度 V 排除黑色，所以先限定 S、V 再切 H，比在 BGR 里卡范围稳。
//   - red 的色相在 0/180 两端（红色一端接 180 一端接 0），必须用两段
//     inRange 再按位或，只取 0~10 会漏掉偏洋红的那一半。
// 每种颜色统计像素数和占比，最后按面积从大到小排序输出，并把每一类
// 单独存成一张二值掩码图。
// ============================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

struct ColorSpec {
    string name;
    vector<pair<int, int>> hueRanges;   // 色相区间（可能两段）
    Scalar bgr;                          // 标注用的颜色
};

int main(int argc, char** argv) {
    string inPath = (argc > 1) ? argv[1] : "images/q08_test.png";
    Mat img = imread(inPath);
    if (img.empty()) {
        cerr << "无法读取图像: " << inPath << endl;
        return 1;
    }

    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    const int SMIN = 60, VMIN = 60;
    vector<ColorSpec> specs = {
        // inRange 的两个端点都算在区间内，所以端点只能归给一种颜色：
        // yellow/green 的 35、green/blue 的 85、blue/purple 的 130、
        // purple/red 的 170 都只写在一边，五个掩码才是互斥的。
        {"red",    {{0, 10}, {170, 179}}, Scalar(0, 0, 255)},
        {"yellow", {{15, 34}},            Scalar(0, 255, 255)},
        {"green",  {{35, 84}},            Scalar(0, 255, 0)},
        {"blue",   {{85, 129}},           Scalar(255, 0, 0)},
        {"purple", {{130, 169}},          Scalar(255, 0, 255)},
    };

    double total = (double)img.rows * img.cols;
    struct Stat { string name; int area; double pct; Scalar color; };
    vector<Stat> stats;
    Mat out = img.clone();

    for (auto& sp : specs) {
        Mat mask = Mat::zeros(img.size(), CV_8U);
        for (auto& hr : sp.hueRanges) {
            Mat part;
            inRange(hsv, Scalar(hr.first, SMIN, VMIN),
                         Scalar(hr.second, 255, 255), part);
            mask |= part;
        }

        // 3x3 开运算去掉边缘的抗锯齿碎点（亚像素混色会产生孤立的
        // 单像素，会让面积统计虚高一点点）
        morphologyEx(mask, mask, MORPH_OPEN,
                     getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

        int area = countNonZero(mask);
        if (area == 0) continue;
        double pct = area / total * 100.0;
        stats.push_back({sp.name, area, pct, sp.bgr});

        // 每类单独存一张二值掩码图
        string fname = "q08_mask_" + sp.name + ".png";
        imwrite(fname, mask);
        cout << "已保存掩码: " << fname << endl;

        // 在原图上把这类区域的轮廓描出来并标注颜色名
        vector<vector<Point>> cc;
        findContours(mask, cc, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (auto& c : cc) {
            if (contourArea(c) < 100) continue;
            drawContours(out, vector<vector<Point>>{c}, 0, sp.bgr, 2);
            Rect r = boundingRect(c);
            putText(out, sp.name, r.tl() + Point(3, 16),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 3, LINE_AA);
            putText(out, sp.name, r.tl() + Point(3, 16),
                    FONT_HERSHEY_SIMPLEX, 0.5, sp.bgr, 1, LINE_AA);
        }
    }

    if (stats.empty()) {
        cerr << "没有检测到任何颜色区域" << endl;
        return 2;
    }

    // ---------- 按面积从大到小排序 ----------
    sort(stats.begin(), stats.end(),
         [](const Stat& a, const Stat& b) { return a.area > b.area; });

    cout << "\n颜色区域面积统计（按面积降序）：" << endl;
    cout << "  排名  颜色     面积(px²)   占比" << endl;
    for (size_t i = 0; i < stats.size(); ++i) {
        char buf[128];
        snprintf(buf, sizeof(buf), "  %-4d  %-7s  %8d   %5.2f%%",
                 (int)i + 1, stats[i].name.c_str(), stats[i].area, stats[i].pct);
        cout << buf << endl;
    }

    // ---------- 图上写总表 ----------
    int y = 28;
    putText(out, "Color area percentage:", Point(10, y),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 3, LINE_AA);
    putText(out, "Color area percentage:", Point(10, y),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1, LINE_AA);
    for (size_t i = 0; i < stats.size(); ++i) {
        y += 26;
        char buf[128];
        snprintf(buf, sizeof(buf), "%s: %d px2 (%.2f%%)",
                 stats[i].name.c_str(), stats[i].area, stats[i].pct);
        putText(out, buf, Point(10, y), FONT_HERSHEY_SIMPLEX, 0.6,
                Scalar(0, 0, 0), 3, LINE_AA);
        putText(out, buf, Point(10, y), FONT_HERSHEY_SIMPLEX, 0.6,
                stats[i].color, 1, LINE_AA);
    }

    string outPath = (argc > 2) ? argv[2] : "q08_result.png";
    imwrite(outPath, out);
    cout << "\n结果图已保存: " << outPath << endl;
    return 0;
}
