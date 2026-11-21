// ============================================================
// 第四题：直线检测与角度测量
//
// 思路：
//   1. 图里线条是纯黑、背景纯白，先阈值化拿到线条掩码。
//   2. 中心点 P 用一个红色小圆点标出，按 HSV 里的红色范围取出它的
//      质心，作为所有角度的原点。
//   3. HoughLinesP 检出直线段。每一段都以"离 P 较远的那个端点"确定
//      方向，用 atan2 求角度；角度统一换算到 [0, 360) 度，按水平向右
//      （P→A 的方向）为 0 度基准。
//   4. 同一条线会被 HoughLinesP 拆成好几段，所以按角度聚类，相差 3
//      度以内的合并成一条射线，再按角度排序输出。
// ============================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

static double normAngle(double deg) {
    while (deg < 0) deg += 360.0;
    while (deg >= 360.0) deg -= 360.0;
    return deg;
}

int main(int argc, char** argv) {
    string inPath = (argc > 1) ? argv[1] : "images/q04_test.png";
    Mat img = imread(inPath);
    if (img.empty()) {
        cerr << "无法读取图像: " << inPath << endl;
        return 1;
    }

    // ---------- 1. 找中心点 P（红色圆点） ----------
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    Mat redMask1, redMask2, redMask;
    inRange(hsv, Scalar(0, 120, 120), Scalar(10, 255, 255), redMask1);
    inRange(hsv, Scalar(170, 120, 120), Scalar(180, 255, 255), redMask2);
    redMask = redMask1 | redMask2;

    Point P(-1, -1);
    Moments rm = moments(redMask, true);
    if (rm.m00 > 0) {
        P = Point((int)(rm.m10 / rm.m00), (int)(rm.m01 / rm.m00));
    } else {
        cerr << "没找到中心点 P" << endl;
        return 2;
    }

    // ---------- 2. 线条掩码（排除红色圆点） ----------
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat bin;
    threshold(gray, bin, 100, 255, THRESH_BINARY_INV);  // 黑线变白
    bin.setTo(0, redMask);                              // 抠掉红点

    // ---------- 3. 概率霍夫变换 ----------
    vector<Vec4i> lines;
    HoughLinesP(bin, lines, 1, CV_PI / 180, 80,
                80,    // minLineLength
                20);   // maxLineGap

    // ---------- 4. 判断每段属于哪条过 P 的射线 ----------
    // 不能要求线段的端点正好落在 P 上：HoughLinesP 会把一条长线拆成
    // 好几截，大部分小段根本不经过 P。正确的做法是看"整条直线"离 P
    // 有多远——把 P 投影到线段所在直线上，垂距小于 12 像素就认为这条
    // 线穿过 P。
    vector<double> angles;
    vector<Vec4i> kept;
    for (const auto& l : lines) {
        Point2d a(l[0], l[1]), b(l[2], l[3]);
        Point2d d = b - a;
        double len2 = d.x * d.x + d.y * d.y;
        if (len2 < 1.0) continue;

        // P 在直线上的投影参数 t
        double t = ((P.x - a.x) * d.x + (P.y - a.y) * d.y) / len2;
        Point2d proj(a.x + t * d.x, a.y + t * d.y);
        if (norm(proj - Point2d(P.x, P.y)) > 12.0) continue;  // 不过 P

        auto angleOf = [&](const Point2d& q) {
            return normAngle(atan2(-(q.y - P.y), (q.x - P.x)) * 180.0 / CV_PI);
        };

        if (t > 0 && t < 1) {
            // P 落在线段内部：这条线往两个方向都延伸，两个方向都算
            angles.push_back(angleOf(a));
            angles.push_back(angleOf(b));
        } else {
            // P 在线段外面：只算指向该线段的那一个方向
            Point2d far = (t <= 0) ? a : b;
            angles.push_back(angleOf(far));
        }
        kept.push_back(l);
    }

    // ---------- 5. 按角度聚类 ----------
    // 先排序，再把相邻间隔 <= 12 度的角度并成一组。用"排序后按间隔分组"
    // 而不是"逐个找最近的簇"，是因为后者对簇的初始值敏感，同一个方向
    // 会被拆成好几簇（实测 219/225/231 会被拆成三条）。
    sort(angles.begin(), angles.end());
    vector<pair<double, int>> acc;   // (组内平均角度, 该组的段数)
    for (double a : angles) {
        if (!acc.empty() && a - acc.back().first <= 12.0) {
            auto& g = acc.back();
            g.first = (g.first * g.second + a) / (g.second + 1);
            ++g.second;
        } else {
            acc.push_back({a, 1});
        }
    }
    // 角度是循环的：359 度和 1 度其实是同一个方向，排序后被分到了
    // 数组首尾。这里检查一下要不要把首尾两组合并。
    if (acc.size() > 1) {
        double wrapGap = (360.0 - acc.back().first) + acc.front().first;
        if (wrapGap <= 12.0) {
            int n = acc.front().second + acc.back().second;
            double avg = ((acc.front().first + 360.0) * acc.front().second +
                          acc.back().first * acc.back().second) / n;
            acc.pop_back();
            acc.front().first = normAngle(avg);
            acc.front().second = n;
        }
    }

    // ---------- 6. 画图并输出 ----------
    Mat out = img.clone();
    circle(out, P, 5, Scalar(0, 0, 255), -1);

    // 找最接近 0 度的那条作为基准射线 PA
    double baseAngle = 0.0;
    if (!acc.empty()) {
        baseAngle = acc.front().first;
        // 实际上 PA 就是水平向右，如果精度不够会在这里偏移，输出时说明
    }

    for (size_t i = 0; i < acc.size(); ++i) {
        double ang = acc[i].first;
        double rad = ang * CV_PI / 180.0;
        Point end((int)(P.x + 300 * cos(rad)),
                  (int)(P.y - 300 * sin(rad)));
        line(out, P, end, Scalar(0, 200, 0), 2, LINE_AA);

        char buf[64];
        snprintf(buf, sizeof(buf), "%.0f deg", ang);
        Point txt((int)(P.x + 170 * cos(rad)),
                  (int)(P.y - 170 * sin(rad)));
        putText(out, buf, txt, FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 0, 0), 3, LINE_AA);
        putText(out, buf, txt, FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 200, 0), 1, LINE_AA);
    }

    cout << "中心点 P: (" << P.x << ", " << P.y << ")" << endl;
    cout << "HoughLinesP 检出 " << lines.size() << " 段，有效 " << kept.size()
         << " 段，聚类成 " << acc.size() << " 条射线" << endl;
    cout << "以水平向右为 0 度基准：" << endl;
    for (size_t i = 0; i < acc.size(); ++i) {
        cout << "  射线 " << (char)('A' + i) << ": " << (int)lround(acc[i].first)
             << " 度（" << acc[i].second << " 段支持）" << endl;
    }
    (void)baseAngle;

    string outPath = (argc > 2) ? argv[2] : "q04_result.png";
    imwrite(outPath, out);
    cout << "结果图已保存: " << outPath << endl;
    return 0;
}
