// ============================================================
// 第一题：多形状物块检测与分类计数
//
// 这张图里的物块会互相压盖：颜色不同的两块叠在一起，在灰度/二值图里
// 就粘成一个连通域，直接 findContours 会把三四个物块数成一个。
//
// 分工是这样的：
//   - 颜色只用来"拆分粘连"。每个物块是单一纯色，所以按色相把前景
//     分层，压在下面的物块虽然被啃掉一块，至少不会和压它的那块混在
//     一起。色相层是从直方图里自动找出来的，没有写死任何一种颜色。
//   - 形状分类完全不看颜色，只用几何量：圆度 4πA/P²、多边形逼近的
//     顶点数、外接矩形的填充率。题目要求的"不能通过颜色判断"指的
//     就是这个分类环节。
//
// 被压盖的物块有缺口，圆度、填充率都会掉下来，这时改用它的凸包来
// 判断：缺口是另一个物块的直边切出来的，所以凸包把缺口补回去以后，
// 圆被补成圆、三角形被补成三角形。
// ============================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

// 一个轮廓的几何特征
struct Feat {
    double area, peri, circ, extent;
    int vtx;
};

static Feat measure(const Mat& mask8) {
    Feat f{0, 0, 0, 0, 0};
    vector<vector<Point>> cs;
    findContours(mask8, cs, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (cs.empty()) return f;
    int bi = 0;
    for (size_t i = 1; i < cs.size(); ++i)
        if (contourArea(cs[i]) > contourArea(cs[bi])) bi = (int)i;

    f.area = contourArea(cs[bi]);
    f.peri = arcLength(cs[bi], true);
    f.circ = (f.peri > 1e-6) ? 4.0 * CV_PI * f.area / (f.peri * f.peri) : 0.0;

    double eps = 0.02 * f.peri;
    vector<Point> approx;
    approxPolyDP(cs[bi], approx, eps, true);
    f.vtx = (int)approx.size();

    Rect br = boundingRect(cs[bi]);
    f.extent = (br.width > 0 && br.height > 0) ? f.area / (double)(br.width * br.height) : 0.0;
    return f;
}

// 凸包版的特征：用来处理被啃掉一块的物块
static Mat hullMask(const Mat& mask8) {
    vector<vector<Point>> cs;
    findContours(mask8, cs, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat m = Mat::zeros(mask8.size(), CV_8U);
    if (cs.empty()) return m;
    int bi = 0;
    for (size_t i = 1; i < cs.size(); ++i)
        if (contourArea(cs[i]) > contourArea(cs[bi])) bi = (int)i;
    vector<Point> hull;
    convexHull(cs[bi], hull);
    drawContours(m, vector<vector<Point>>{hull}, -1, Scalar(255), FILLED);
    return m;
}

// 兜底判据：拿三种外接模型（最小外接圆 / 最小外接矩形 / 最小外接
// 三角形）分别去覆盖轮廓，哪个模型被填得最满就判成哪一类。物块只剩
// 一个楔形、凸包也判断不出形状时用这一招。
static int bestFitModel(const Mat& mask8) {
    vector<vector<Point>> cs;
    findContours(mask8, cs, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (cs.empty()) return 2;
    int bi = 0;
    for (size_t i = 1; i < cs.size(); ++i)
        if (contourArea(cs[i]) > contourArea(cs[bi])) bi = (int)i;

    double a = contourArea(cs[bi]);
    if (a <= 1) return 2;

    Point2f c;
    float r;
    minEnclosingCircle(cs[bi], c, r);
    double rc = (r > 0) ? a / (CV_PI * r * r) : 0;

    RotatedRect rr = minAreaRect(cs[bi]);
    double ra = (double)rr.size.width * rr.size.height;
    double rrect = (ra > 0) ? a / ra : 0;

    vector<Point2f> tri;
    minEnclosingTriangle(cs[bi], tri);
    double ta = 0;
    if (tri.size() == 3) ta = fabs(contourArea(tri));
    double rtri = (ta > 0) ? a / ta : 0;

    if (rc >= rrect && rc >= rtri) return 0;   // 圆
    if (rtri > rrect) return 1;                // 三角形
    return 2;                                  // 矩形
}

// 色相直方图的峰：前景像素按色相投票，票数够多的局部峰就是一个
// 颜色层。相邻峰之间取中点作为分界。
static vector<int> findHuePeaks(const Mat& H, const Mat& fg) {
    int hist[180] = {0};
    for (int y = 0; y < fg.rows; ++y)
        for (int x = 0; x < fg.cols; ++x)
            if (fg.at<uchar>(y, x)) ++hist[H.at<uchar>(y, x)];

    long long total = 0;
    for (int i = 0; i < 180; ++i) total += hist[i];
    if (total == 0) return {};

    // 3 点平滑后找局部极大，阈值取总票数的 3%
    int thr = (int)(total * 0.03);
    vector<int> peaks;
    for (int i = 0; i < 180; ++i) {
        int a = hist[(i + 179) % 180], b = hist[i], c = hist[(i + 1) % 180];
        if (b >= thr && b >= a && b > c) peaks.push_back(i);
    }
    // 相邻的峰（间距 < 6）合并成一个
    vector<int> merged;
    for (int p : peaks) {
        if (!merged.empty()) {
            int q = merged.back();
            int d = abs(p - q);
            d = min(d, 180 - d);
            if (d < 6) {
                if (hist[p] > hist[q]) merged.back() = p;
                continue;
            }
        }
        merged.push_back(p);
    }
    return merged;
}

int main(int argc, char** argv) {
    string inPath = (argc > 1) ? argv[1] : "images/q01_test.png";
    Mat img = imread(inPath);
    if (img.empty()) {
        cerr << "无法读取图像: " << inPath << endl;
        return 1;
    }

    // ---------- 1. 前景掩码：只靠饱和度，不看色相 ----------
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    Mat Hh, Ss, Vv;
    vector<Mat> ch;
    split(hsv, ch);
    Hh = ch[0]; Ss = ch[1]; Vv = ch[2];

    Mat fg;
    inRange(hsv, Scalar(0, 90, 60), Scalar(180, 255, 255), fg);

    // ---------- 2. 按色相分层（只为了拆分粘连，不参与分类） ----------
    vector<int> peaks = findHuePeaks(Hh, fg);
    if (peaks.empty()) {
        cerr << "没有找到任何颜色层" << endl;
        return 2;
    }

    Mat layer(img.size(), CV_8U, Scalar(0));   // 0 背景，1..P 层号
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) {
            if (!fg.at<uchar>(y, x)) continue;
            int h = Hh.at<uchar>(y, x);
            int best = 0, bd = 999;
            for (size_t k = 0; k < peaks.size(); ++k) {
                int d = abs(h - peaks[k]);
                d = min(d, 180 - d);
                if (d < bd) { bd = d; best = (int)k; }
            }
            layer.at<uchar>(y, x) = (uchar)(best + 1);
        }

    // ---------- 3. 逐层分类 ----------
    int nCircle = 0, nRect = 0, nTri = 0, nOther = 0;
    Mat out = img.clone();
    Mat k3 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));

    const Scalar colCircle(0, 255, 0);
    const Scalar colRect(255, 0, 0);
    const Scalar colTri(0, 0, 255);
    const Scalar colOther(0, 165, 255);

    for (size_t k = 0; k < peaks.size(); ++k) {
        Mat one = (layer == (int)k + 1);
        one.convertTo(one, CV_8U, 255);
        morphologyEx(one, one, MORPH_OPEN, k3);

        Mat lbl, stats, cents;
        int n = connectedComponentsWithStats(one, lbl, stats, cents);

        for (int i = 1; i < n; ++i) {
            int area = stats.at<int>(i, CC_STAT_AREA);
            if (area < 200) continue;              // 抗锯齿碎边

            Mat piece = (lbl == i);
            piece.convertTo(piece, CV_8U, 255);

            Feat f = measure(piece);
            string label;
            Scalar color;

            if (f.circ >= 0.85) {                   // 完整的圆
                ++nCircle; label = "C" + to_string(nCircle); color = colCircle;
            } else if (f.vtx == 3 && f.extent < 0.60) {   // 完整的三角形
                ++nTri; label = "T" + to_string(nTri); color = colTri;
            } else if (f.vtx == 4 && f.extent >= 0.90) {  // 完整的矩形
                ++nRect; label = "R" + to_string(nRect); color = colRect;
            } else {
                // 被压盖 / 和同色物块粘连：用凸包把缺口补回去再判断
                Mat hull = hullMask(piece);
                Feat h = measure(hull);
                if (h.circ >= 0.82) {
                    ++nCircle; label = "C" + to_string(nCircle); color = colCircle;
                } else if (h.vtx == 3) {
                    ++nTri; label = "T" + to_string(nTri); color = colTri;
                } else if (h.vtx == 4 || (h.vtx <= 6 && h.extent > 0.75)) {
                    ++nRect; label = "R" + to_string(nRect); color = colRect;
                } else {
                    // 只剩一个楔形时凸包也认不出来，用外接模型兜底
                    int mt = bestFitModel(hull);
                    if (mt == 0) { ++nCircle; label = "C" + to_string(nCircle); color = colCircle; }
                    else if (mt == 1) { ++nTri; label = "T" + to_string(nTri); color = colTri; }
                    else { ++nRect; label = "R" + to_string(nRect); color = colRect; }
                }
            }

            // 用凸包画框：被压盖的物块也能框出完整范围
            Mat hull = hullMask(piece);
            vector<vector<Point>> cc;
            findContours(hull, cc, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            if (!cc.empty()) drawContours(out, cc, -1, color, 2);

            Moments m = moments(cc.empty() ? piece : hull, true);
            if (m.m00 > 0) {
                Point c((int)(m.m10 / m.m00), (int)(m.m01 / m.m00));
                putText(out, label, c + Point(-10, 5), FONT_HERSHEY_SIMPLEX, 0.5,
                        Scalar(0, 0, 0), 3, LINE_AA);
                putText(out, label, c + Point(-10, 5), FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, LINE_AA);
            }
        }
    }

    // ---------- 4. 输出 ----------
    cout << "颜色层数: " << peaks.size() << "（色相峰值";
    for (size_t k = 0; k < peaks.size(); ++k) cout << " " << peaks[k];
    cout << "）" << endl;
    cout << "圆形: " << nCircle << " 个" << endl;
    cout << "矩形: " << nRect << " 个" << endl;
    cout << "三角形: " << nTri << " 个" << endl;
    if (nOther) cout << "未分类: " << nOther << " 个" << endl;
    cout << "合计: " << nCircle + nRect + nTri + nOther << " 个" << endl;

    string head = "Circles:" + to_string(nCircle) + "  Rects:" + to_string(nRect) +
                  "  Triangles:" + to_string(nTri);
    putText(out, head, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8,
            Scalar(0, 0, 0), 3, LINE_AA);
    putText(out, head, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8,
            Scalar(0, 0, 255), 1, LINE_AA);

    string outPath = (argc > 2) ? argv[2] : "q01_result.png";
    imwrite(outPath, out);
    cout << "结果图已保存: " << outPath << endl;
    return 0;
}
