// ============================================================
// 第十题：多目标轨迹追踪与分析
//
// 图里三条轨迹分别用红、蓝、绿画，每种颜色还写了 "A--start" 之类的小字
// 标签，所以先按连通域面积把文字滤掉，只留下轨迹本身。
//
// 分析步骤：
//   1. HSV 阈值分出三种颜色，丢掉面积 < 100 的小连通域（文字标签）。
//   2. 定"主方向"：比较 x、y 两个方向的方差，哪个方向铺得开就把哪个
//      当自变量。红、蓝两条是竖着走的，按 y 取点；绿色是横着走的，
//      按 x 取点。
//   3. 沿主轴做分箱（每 4 像素一箱），每箱取另坐标的均值，得到一条
//      单值曲线。这样做有两个好处：去掉了线宽带来的抖动，也让三条
//      轨迹被绿色遮挡、断成几段时还能接成一条完整曲线。
//   4. 用一、二次多项式分别最小二乘拟合，比较残差 RMSE：
//        直线拟合就够准        -> 直线
//        二次拟合明显更准      -> 抛物线
//        两种都拟合不像，但去掉直线趋势后反复上下穿过零线 -> 正弦
//   5. 轨迹长度 = 分箱后的折线长度之和；正弦再额外报振幅和周期。
// ============================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

// 多项式最小二乘拟合：u 归一化到 [0,1] 再解正规方程，避免 u² 量级
// 太大导致的病态矩阵。返回 RMSE，可选带回系数。
static double polyFitRMSE(const vector<double>& U, const vector<double>& V,
                          int deg, vector<double>* coefOut = nullptr) {
    int N = (int)U.size();
    int d = deg + 1;
    double u0 = U.front(), du = U.back() - U.front();
    if (du < 1e-9) du = 1.0;

    vector<vector<double>> A(d, vector<double>(d, 0.0));
    vector<double> B(d, 0.0);
    for (int i = 0; i < N; ++i) {
        double u = (U[i] - u0) / du;
        vector<double> pw(d, 1.0);
        for (int k = 1; k < d; ++k) pw[k] = pw[k - 1] * u;
        for (int a = 0; a < d; ++a) {
            B[a] += pw[a] * V[i];
            for (int b = 0; b < d; ++b) A[a][b] += pw[a] * pw[b];
        }
    }

    // 高斯消元（列主元）
    for (int i = 0; i < d; ++i) {
        int piv = i;
        for (int k = i + 1; k < d; ++k)
            if (fabs(A[k][i]) > fabs(A[piv][i])) piv = k;
        swap(A[i], A[piv]);
        swap(B[i], B[piv]);
        for (int k = i + 1; k < d; ++k) {
            double f = A[k][i] / A[i][i];
            for (int j = i; j < d; ++j) A[k][j] -= f * A[i][j];
            B[k] -= f * B[i];
        }
    }
    vector<double> co(d, 0.0);
    for (int i = d - 1; i >= 0; --i) {
        double s = B[i];
        for (int j = i + 1; j < d; ++j) s -= A[i][j] * co[j];
        co[i] = s / A[i][i];
    }
    if (coefOut) *coefOut = co;

    double se = 0;
    for (int i = 0; i < N; ++i) {
        double u = (U[i] - u0) / du;
        double pw = 1, y = 0;
        for (int k = 0; k < d; ++k) { y += co[k] * pw; pw *= u; }
        se += (V[i] - y) * (V[i] - y);
    }
    return sqrt(se / N);
}

struct Track {
    string name;
    Scalar color;
    vector<Point2d> prof;     // 分箱后的折线点 (u, v)
    bool sortX;
    double length;
    double lineRMSE, paraRMSE;
    int crossings;
    double amp, period;       // 正弦的振幅、周期（像素）
    int peaks;                // 波峰个数
    string type, detail;
    Point2d start, finish;
    double hspan, vspan;
};

int main(int argc, char** argv) {
    string inPath = (argc > 1) ? argv[1] : "images/q10_test.png";
    Mat img = imread(inPath);
    if (img.empty()) {
        cerr << "无法读取图像: " << inPath << endl;
        return 1;
    }

    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    struct Spec { string name; int h1, h2; Scalar bgr; };
    vector<Spec> specs = {
        {"A(红色)", 0,   10,  Scalar(0, 0, 255)},
        {"B(蓝色)", 100, 130, Scalar(255, 0, 0)},
        {"C(绿色)", 35,  85,  Scalar(0, 200, 0)},
    };

    Mat out = img.clone();
    vector<Track> tracks;

    for (auto& sp : specs) {
        Mat m;
        inRange(hsv, Scalar(sp.h1, 80, 80), Scalar(sp.h2, 255, 255), m);

        // ---------- 1. 去掉文字标签（小连通域） ----------
        Mat lbl, stats, cents;
        int n = connectedComponentsWithStats(m, lbl, stats, cents);
        Mat keep = Mat::zeros(m.size(), CV_8U);
        for (int i = 1; i < n; ++i)
            if (stats.at<int>(i, CC_STAT_AREA) >= 100)
                keep.setTo(255, (lbl == i));

        vector<Point> pts;
        findNonZero(keep, pts);
        if (pts.size() < 50) {
            cerr << sp.name << " 轨迹点太少，跳过" << endl;
            continue;
        }

        // ---------- 2. 定主方向 ----------
        double mx = 0, my = 0;
        for (const auto& p : pts) { mx += p.x; my += p.y; }
        mx /= pts.size(); my /= pts.size();
        double sxx = 0, syy = 0;
        for (const auto& p : pts) {
            double dx = p.x - mx, dy = p.y - my;
            sxx += dx * dx; syy += dy * dy;
        }
        bool sortX = (sxx >= syy);

        // ---------- 3. 分箱平滑 ----------
        const int BIN = 4;
        vector<vector<double>> bins(10000);
        for (const auto& p : pts) {
            int u = sortX ? p.x : p.y;
            int v = sortX ? p.y : p.x;
            int b = u / BIN;
            if (b < 0 || b >= 10000) continue;
            bins[b].push_back((double)v);
        }
        vector<Point2d> prof;   // (u, v)
        for (int b = 0; b < 10000; ++b) {
            if (bins[b].empty()) continue;
            double s = 0;
            for (double v : bins[b]) s += v;
            prof.push_back(Point2d(b * BIN + BIN / 2.0, s / bins[b].size()));
        }
        if (prof.size() < 10) continue;

        Track t;
        t.name = sp.name;
        t.color = sp.bgr;
        t.sortX = sortX;
        t.prof = prof;

        // ---------- 4. 长度与跨度 ----------
        double len = 0;
        for (size_t i = 1; i < prof.size(); ++i) len += norm(prof[i] - prof[i - 1]);
        t.length = len;
        double umin = 1e9, umax = -1e9, vmin = 1e9, vmax = -1e9;
        for (auto& p : prof) {
            umin = min(umin, p.x); umax = max(umax, p.x);
            vmin = min(vmin, p.y); vmax = max(vmax, p.y);
        }
        t.hspan = sortX ? (umax - umin) : (vmax - vmin);
        t.vspan = sortX ? (vmax - vmin) : (umax - umin);

        // ---------- 5. 拟合 ----------
        vector<double> U, V;
        for (auto& p : prof) { U.push_back(p.x); V.push_back(p.y); }
        t.lineRMSE = polyFitRMSE(U, V, 1);
        t.paraRMSE = polyFitRMSE(U, V, 2);

        // 去直线趋势后的残差：用来判断正弦
        vector<double> co1;
        polyFitRMSE(U, V, 1, &co1);
        double u0 = U.front(), du = U.back() - U.front();
        if (du < 1e-9) du = 1.0;
        vector<double> res;
        for (size_t i = 0; i < U.size(); ++i) {
            double u = (U[i] - u0) / du;
            res.push_back(V[i] - (co1[0] + co1[1] * u));
        }
        // 三点滑动平均，防止噪声制造假过零
        vector<double> sm = res;
        for (size_t i = 1; i + 1 < res.size(); ++i)
            sm[i] = (res[i - 1] + res[i] + res[i + 1]) / 3.0;

        int cross = 0;
        vector<double> crossPos;
        for (size_t i = 1; i < sm.size(); ++i)
            if ((sm[i - 1] < 0 && sm[i] >= 0) || (sm[i - 1] >= 0 && sm[i] < 0)) {
                ++cross;
                crossPos.push_back(prof[i].x);
            }
        t.crossings = cross;

        // 振幅 = 残差最大绝对值；周期 = 相邻过零点间距的两倍
        double amp = 0;
        for (double r : sm) amp = max(amp, fabs(r));
        t.amp = amp;
        t.period = 0;
        if (crossPos.size() >= 2)
            t.period = 2.0 * (crossPos.back() - crossPos.front()) / (crossPos.size() - 1);

        // 波峰数：残差由升转降的次数
        int peaks = 0;
        for (size_t i = 1; i + 1 < sm.size(); ++i)
            if (sm[i] > sm[i - 1] && sm[i] >= sm[i + 1] && sm[i] > 0.3 * amp) ++peaks;
        t.peaks = peaks;

        // ---------- 6. 判定运动类型 ----------
        if (t.lineRMSE < 3.0) {
            t.type = "直线运动";
            t.detail = "一阶拟合残差只有 " + to_string((int)lround(t.lineRMSE)) + " px";
        } else if (cross >= 4 && t.paraRMSE > 0.6 * t.lineRMSE) {
            t.type = "正弦曲线运动";
            t.detail = "去直线趋势后 " + to_string(cross) + " 次穿过零线，振幅约 " +
                       to_string((int)lround(t.amp)) + " px，周期约 " +
                       to_string((int)lround(t.period)) + " px";
        } else if (t.paraRMSE < 0.5 * t.lineRMSE) {
            t.type = "抛物线运动";
            t.detail = "二次拟合残差 " + to_string((int)lround(t.paraRMSE)) +
                       " px，远小于直线拟合的 " + to_string((int)lround(t.lineRMSE)) + " px";
        } else {
            t.type = "一般曲线运动";
            t.detail = "直线拟合残差 " + to_string((int)lround(t.lineRMSE)) + " px";
        }

        t.start = sortX ? Point2d(prof.front().x, prof.front().y)
                        : Point2d(prof.front().y, prof.front().x);
        t.finish = sortX ? Point2d(prof.back().x, prof.back().y)
                         : Point2d(prof.back().y, prof.back().x);
        tracks.push_back(t);
    }

    if (tracks.empty()) {
        cerr << "没有检测到任何轨迹" << endl;
        return 2;
    }

    // ---------- 7. 画图 ----------
    for (auto& t : tracks) {
        // 分箱折线：先画黑边再画彩线，叠在原轨迹上看得清
        for (size_t i = 1; i < t.prof.size(); ++i) {
            Point2d a = t.prof[i - 1], b = t.prof[i];
            Point pa = t.sortX ? Point((int)a.x, (int)a.y) : Point((int)a.y, (int)a.x);
            Point pb = t.sortX ? Point((int)b.x, (int)b.y) : Point((int)b.y, (int)b.x);
            line(out, pa, pb, Scalar(0, 0, 0), 3, LINE_AA);
            line(out, pa, pb, t.color, 1, LINE_AA);
        }
        circle(out, Point((int)t.start.x, (int)t.start.y), 5, t.color, 2);
        circle(out, Point((int)t.finish.x, (int)t.finish.y), 5, t.color, 2);

        char buf[160];
        snprintf(buf, sizeof(buf), "%s %s L=%dpx", t.name.c_str(),
                 t.type.c_str(), (int)lround(t.length));
        Point tp((int)t.start.x - 20, (int)t.start.y - 12);
        putText(out, buf, tp, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 3, LINE_AA);
        putText(out, buf, tp, FONT_HERSHEY_SIMPLEX, 0.5, t.color, 1, LINE_AA);
    }

    char head[96];
    snprintf(head, sizeof(head), "Objects: %d", (int)tracks.size());
    putText(out, head, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7,
            Scalar(0, 0, 0), 3, LINE_AA);
    putText(out, head, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7,
            Scalar(255, 255, 255), 1, LINE_AA);

    // ---------- 8. 控制台报告 ----------
    cout << "检测到轨迹数量: " << tracks.size() << endl;
    for (auto& t : tracks) {
        cout << "\n" << t.name << " 轨迹" << endl;
        cout << "  起点 (" << (int)t.start.x << ", " << (int)t.start.y << ")"
             << "  终点 (" << (int)t.finish.x << ", " << (int)t.finish.y << ")" << endl;
        cout << "  运动类型: " << t.type << "（" << t.detail << "）" << endl;
        cout << "  轨迹总长度: " << (int)lround(t.length) << " px" << endl;
        cout << "  跨度: 水平 " << (int)lround(t.hspan)
             << " px，竖直 " << (int)lround(t.vspan) << " px" << endl;
        if (t.type.find("正弦") != string::npos)
            cout << "  波峰 " << t.peaks << " 个，波峰波谷落差约 "
                 << (int)lround(2 * t.amp) << " px" << endl;
    }

    // 按长度排序，说明谁最长
    vector<int> idx(tracks.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
    sort(idx.begin(), idx.end(),
         [&](int a, int b) { return tracks[a].length > tracks[b].length; });

    cout << "\n轨迹长度排序: ";
    for (size_t i = 0; i < idx.size(); ++i) {
        cout << (i ? " > " : "") << tracks[idx[i]].name << "("
             << (int)lround(tracks[idx[i]].length) << ")";
    }
    cout << endl;

    const Track& top = tracks[idx[0]];
    cout << "轨迹最长的物体: " << top.name << "，" << (int)lround(top.length) << " px" << endl;
    if (idx.size() > 1) {
        const Track& second = tracks[idx[1]];
        double diff = top.length - second.length;
        if (diff < 0.1 * top.length)
            cout << "  （与第二长 " << second.name << " 只差 "
                 << (int)lround(diff) << " px，两者几乎等长）" << endl;
    }
    cout << "竖直落差最大的是 ";
    int iv = 0;
    for (size_t i = 1; i < tracks.size(); ++i)
        if (tracks[i].vspan > tracks[iv].vspan) iv = (int)i;
    cout << tracks[iv].name << "（" << (int)lround(tracks[iv].vspan) << " px）" << endl;

    string outPath = (argc > 2) ? argv[2] : "q10_result.png";
    imwrite(outPath, out);
    cout << "结果图已保存: " << outPath << endl;
    return 0;
}
