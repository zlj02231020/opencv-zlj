// ============================================================
// 第七题：焊缝裂纹检测
//
// 图是暗色金属板，裂纹是一两条像素宽的细黑线，和背景的灰度差很小
// （实测残差最大只有 50 左右，大部分裂纹像素差 12~25），所以：
//   1. 背景估计用大核中值滤波把细线抹掉，再用 背景-原图 得到残差，
//      这样只跟"局部比周围暗多少"有关，跟整张图的亮度无关。
//   2. 残差阈值 14 二值化。7x7 的闭运算用来接上主裂纹中间断开的一截
//      （阈值一高，裂纹在弯曲处会断成两三段）。
//   3. 每段裂纹用 ximgproc::thinning 细化成单像素骨架，再在骨架上取
//      直径（最长路径）当裂纹的实际走向长度——直接数骨架像素数会把
//      斜着走的像素少算，枝条上的毛刺又多算。8 邻域是带权图（斜向
//      √2），所以用两遍 Dijkstra 求直径，不用普通 BFS。
//   4. 面积 < 40 或被闭运算连不上的碎屑丢掉。
//   5. 每段裂纹用不同颜色描出来，标上编号和长度。
// ============================================================
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <functional>

using namespace std;
using namespace cv;

// 在骨架图上求最长路径（直径）：先从任意骨架点出发找最远点，再从那个
// 点出发找最远点，两次之间的路径就是骨架的直径。
// 8 邻域里横竖一步算 1、斜向一步算 √2，这是个带权图，普通 BFS 按入队
// 顺序扩散、不按距离，带权时结果会偏长；所以这里用 Dijkstra。骨架基本
// 是树，两遍扫描对树是精确的，个别小环只会带来很小的误差。
static double longestPath(const Mat& sk, Point& pa, Point& pb) {
    int H = sk.rows, W = sk.cols;
    const double INF = 1e18;

    auto dijkstra = [&](int sy, int sx, double* outDist, int* ey, int* ex) {
        vector<double> d((size_t)H * W, INF);
        vector<char> done((size_t)H * W, 0);
        using Item = pair<double, int>;                    // (距离, 线性下标)
        priority_queue<Item, vector<Item>, greater<Item>> pq;
        int start = sy * W + sx;
        d[start] = 0;
        pq.push({0.0, start});
        double best = 0;
        int bi = start;
        while (!pq.empty()) {
            double dist = pq.top().first;
            int id = pq.top().second;
            pq.pop();
            if (done[id]) continue;
            done[id] = 1;
            if (dist > best) { best = dist; bi = id; }
            int y = id / W, x = id % W;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (!dx && !dy) continue;
                    int ny = y + dy, nx = x + dx;
                    if (ny < 0 || nx < 0 || ny >= H || nx >= W) continue;
                    if (!sk.at<uchar>(ny, nx)) continue;
                    int nid = ny * W + nx;
                    double nd = dist + ((dx == 0 || dy == 0) ? 1.0 : 1.41421356);
                    if (nd < d[nid]) { d[nid] = nd; pq.push({nd, nid}); }
                }
        }
        *outDist = best; *ey = bi / W; *ex = bi % W;
    };

    int sy = -1, sx = -1;
    for (int y = 0; y < H && sy < 0; ++y)
        for (int x = 0; x < W; ++x)
            if (sk.at<uchar>(y, x)) { sy = y; sx = x; break; }
    if (sy < 0) return 0;

    double d1; int y1, x1;
    dijkstra(sy, sx, &d1, &y1, &x1);
    double d2; int y2, x2;
    dijkstra(y1, x1, &d2, &y2, &x2);

    pa = Point(x1, y1);
    pb = Point(x2, y2);
    return d2;
}

int main(int argc, char** argv) {
    string inPath = (argc > 1) ? argv[1] : "images/q07_test.png";
    Mat img = imread(inPath);
    if (img.empty()) {
        cerr << "无法读取图像: " << inPath << endl;
        return 1;
    }

    // ---------- 1. 残差图 ----------
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat bg;
    medianBlur(gray, bg, 31);          // 大核中值：细线、小斑点全部被抹掉
    Mat diff;
    subtract(bg, gray, diff);          // 正数 = 比周围暗

    // ---------- 2. 二值化 + 接断点 ----------
    Mat bw;
    threshold(diff, bw, 14, 255, THRESH_BINARY);
    bw.convertTo(bw, CV_8U);
    morphologyEx(bw, bw, MORPH_CLOSE,
                 getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));

    // 注意这里不能再做开运算：3x3 的开会把本来就一两像素宽的裂纹
    // 从最细的地方整条切断，主裂纹会碎成两段。噪点靠后面的面积和
    // 骨架长度过滤。

    // ---------- 3. 逐段提取骨架、算长度 ----------
    Mat lbl, stats, cents;
    int n = connectedComponentsWithStats(bw, lbl, stats, cents);

    struct Crack {
        vector<Point> skel;
        double length;
        double chord;      // 首尾直线距离
        int area;
        Point a, b;
    };
    vector<Crack> cracks;

    for (int i = 1; i < n; ++i) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area < 40) continue;               // 碎屑

        Mat one = (lbl == i);
        // thinning 要求输入是 8 位单通道 0/255，且原地修改，先转一份
        Mat bin;
        one.convertTo(bin, CV_8U);
        Mat sk;
        ximgproc::thinning(bin, sk, ximgproc::THINNING_ZHANGSUEN);

        Point a, b;
        double L = longestPath(sk, a, b);
        if (L < 25) continue;                  // 太短的当噪点

        Crack c;
        c.length = L;
        c.chord = norm(a - b);
        c.area = area;
        c.a = a; c.b = b;
        findNonZero(sk, c.skel);
        cracks.push_back(c);
    }

    if (cracks.empty()) {
        cerr << "没有检测到裂纹" << endl;
        return 2;
    }

    // 按长度从大到小排，Crack-1 就是最长那条
    sort(cracks.begin(), cracks.end(),
         [](const Crack& x, const Crack& y) { return x.length > y.length; });

    // ---------- 4. 画图 ----------
    Mat out = img.clone();
    // 原图很暗，先把整体提亮一点，否则画的线也看不清
    out.convertTo(out, -1, 1.8, 40);

    const Scalar colors[3] = {Scalar(0, 0, 255),      // 红：主裂纹
                              Scalar(0, 255, 255),    // 黄
                              Scalar(255, 0, 255)};   // 品红
    for (size_t i = 0; i < cracks.size(); ++i) {
        const Scalar& col = colors[i % 3];
        for (const auto& p : cracks[i].skel)
            circle(out, p, 0, col, -1);

        // 端点画小圈，便于看出首尾
        circle(out, cracks[i].a, 3, col, 1);
        circle(out, cracks[i].b, 3, col, 1);

        Point mid((cracks[i].a.x + cracks[i].b.x) / 2,
                  (cracks[i].a.y + cracks[i].b.y) / 2);
        char buf[64];
        snprintf(buf, sizeof(buf), "Crack-%d L=%d", (int)i + 1,
                 (int)lround(cracks[i].length));
        putText(out, buf, mid + Point(-30, -12), FONT_HERSHEY_SIMPLEX, 0.45,
                Scalar(0, 0, 0), 3, LINE_AA);
        putText(out, buf, mid + Point(-30, -12), FONT_HERSHEY_SIMPLEX, 0.45,
                col, 1, LINE_AA);
    }

    char head[128];
    snprintf(head, sizeof(head), "Cracks Detected: %d", (int)cracks.size());
    putText(out, head, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7,
            Scalar(0, 255, 255), 2, LINE_AA);

    // ---------- 5. 输出 ----------
    cout << "裂纹数量: " << cracks.size() << " 条" << endl;
    for (size_t i = 0; i < cracks.size(); ++i) {
        cout << "  Crack-" << i + 1
             << ": 骨架长度 " << (int)lround(cracks[i].length) << " px"
             << "（首尾直线 " << (int)lround(cracks[i].chord) << " px，"
             << "面积 " << cracks[i].area << " px²）" << endl;
    }
    cout << "最长裂纹: " << (int)lround(cracks[0].length) << " px" << endl;

    string outPath = (argc > 2) ? argv[2] : "q07_result.png";
    imwrite(outPath, out);
    cout << "结果图已保存: " << outPath << endl;
    return 0;
}
