// ============================================================
// 第十题：多目标轨迹追踪与分析
//
// 图里三条轨迹分别用红、蓝、绿画，每种颜色还写了 "A--start" 这类小字
// 标签，所以先按连通域面积把文字滤掉，只留下轨迹本身。
//
// 分析步骤：
//   1. HSV 阈值分出三种颜色，丢掉面积 < 100 的小连通域（文字标签）。
//   2. 定"主方向"：比较 x、y 两个方向的方差，哪个方向铺得开就把哪个
//      当自变量。红、蓝两条是竖着走的，按 y 取点；绿色是横着走的，
//      按 x 取点。
//   3. 沿主轴做分箱（每 4 像素一箱），每箱取另坐标的均值，得到一条
//      单值曲线。这样做有两个好处：去掉了线宽带来的抖动，也让三条
//      轨迹被绿色遮挡、断成几段时还能接成完整曲线。
//   4. 用一、二次多项式分别最小二乘拟合，比较残差 RMSE：
//        直线拟合就够准        -> 直线
//        二次拟合明显更准      -> 抛物线
//        两种都拟合不像，但去掉直线趋势后反复上下穿过零线 -> 正弦
//   5. 轨迹长度 = 分箱后的折线长度之和；正弦再额外报振幅和周期。
//   6. 起点/终点：图上本来就写着 "A--start" 之类的小字，按同色的小
//      连通域把字聚成组，离哪个端点近就是哪个端点的标签；一组里
//      "start"（5 个字母）比 "end"（3 个字母）宽，宽度大的那个是起点。
//      不读标签的话，分箱是从小坐标往大坐标走的，起点会被标反。
//
// 结果图 = 左边原图（只标端点和正弦的振幅/周期）+ 右边一列数据面板。
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
    string name;      // 图上用的短名，必须是 ASCII
    string cn;        // 控制台用的中文名
    string typeEn;    // 图上用的运动类型，必须是 ASCII
    Scalar color;
    vector<Point2d> prof;     // 分箱后的折线点 (u, v)
    bool sortX;
    double length;
    double lineRMSE, paraRMSE;
    int crossings;
    double amp, period;       // 正弦的振幅、峰谷落差、周期（像素）
    double resMin, resMax;
    Point2d pk, tr;           // 检出的波峰、波谷（屏幕坐标）
    int peaks;
    string type, detail;
    Point2d start, finish;
    double hspan, vspan;
    bool startFromLabel;
};

// 把某一颜色的小连通域（文字）按 x 聚成组，返回每组的中心和外接框。
// 用来读图上 "A--start" 这类标签。
struct TextGroup { Point2f c; int w; };
static vector<TextGroup> findTextGroups(const Mat& mask, const Mat& lbl,
                                        const Mat& stats, int n) {
    vector<Rect> blobs;
    for (int i = 1; i < n; ++i) {
        int a = stats.at<int>(i, CC_STAT_AREA);
        int w = stats.at<int>(i, CC_STAT_WIDTH);
        int h = stats.at<int>(i, CC_STAT_HEIGHT);
        // 字形：面积 5~120，单个字不会比 40x25 还大
        if (a >= 5 && a <= 120 && w <= 40 && h <= 25)
            blobs.push_back(Rect(stats.at<int>(i, CC_STAT_LEFT),
                                 stats.at<int>(i, CC_STAT_TOP), w, h));
    }
    sort(blobs.begin(), blobs.end(),
         [](const Rect& x, const Rect& y) { return x.x < y.x; });

    vector<TextGroup> groups;
    vector<Rect> cur;
    auto flush = [&]() {
        if (cur.empty()) return;
        Rect bb = cur[0];
        for (auto& r : cur) bb |= r;
        groups.push_back({Point2f(bb.x + bb.width / 2.0f, bb.y + bb.height / 2.0f),
                          bb.width});
        cur.clear();
    };
    for (auto& r : blobs) {
        if (!cur.empty() && r.x - (cur.back().x + cur.back().width) > 40) flush();
        cur.push_back(r);
    }
    flush();
    return groups;
}

static void putLabel(Mat& m, const string& s, Point p, Scalar col,
                     double sc = 0.45, int th = 1) {
    putText(m, s, p, FONT_HERSHEY_SIMPLEX, sc, Scalar(255, 255, 255), th + 3, LINE_AA);
    putText(m, s, p, FONT_HERSHEY_SIMPLEX, sc, col, th, LINE_AA);
}

int main(int argc, char** argv) {
    string inPath = (argc > 1) ? argv[1] : "images/q10_test.png";
    Mat img = imread(inPath);
    if (img.empty()) {
        cerr << "无法读取图像: " << inPath << endl;
        return 1;
    }

    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    struct Spec { string name; string cn; int h1, h2; Scalar bgr; };
    vector<Spec> specs = {
        {"A", "A(红色)", 0,   10,  Scalar(0, 0, 255)},
        {"B", "B(蓝色)", 100, 130, Scalar(255, 0, 0)},
        {"C", "C(绿色)", 35,  85,  Scalar(0, 200, 0)},
    };

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
            cerr << sp.cn << " 轨迹点太少，跳过" << endl;
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
        t.cn = sp.cn;
        t.color = sp.bgr;
        t.sortX = sortX;
        t.prof = prof;
        t.startFromLabel = false;

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

        // 去直线趋势后的残差：用来判断正弦、量振幅
        vector<double> co1;
        polyFitRMSE(U, V, 1, &co1);
        double u0 = U.front(), du = U.back() - U.front();
        if (du < 1e-9) du = 1.0;
        vector<double> res;
        for (size_t i = 0; i < U.size(); ++i) {
            double u = (U[i] - u0) / du;
            res.push_back(V[i] - (co1[0] + co1[1] * u));
        }
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

        // 峰谷：残差的最大/最小值。振幅取落差的一半，这样对称的正弦
        // 得到的就是真正的半峰谷差。
        t.resMax = -1e9; t.resMin = 1e9;
        size_t iMax = 0, iMin = 0;
        for (size_t i = 0; i < sm.size(); ++i) {
            if (sm[i] > t.resMax) { t.resMax = sm[i]; iMax = i; }
            if (sm[i] < t.resMin) { t.resMin = sm[i]; iMin = i; }
        }
        t.amp = 0.5 * (t.resMax - t.resMin);
        t.period = 0;
        if (crossPos.size() >= 2)
            t.period = 2.0 * (crossPos.back() - crossPos.front()) / (crossPos.size() - 1);

        int peaks = 0;
        for (size_t i = 1; i + 1 < sm.size(); ++i)
            if (sm[i] > sm[i - 1] && sm[i] >= sm[i + 1] && sm[i] > 0.3 * t.amp * 2) ++peaks;
        t.peaks = peaks;

        // 波峰/波谷的屏幕坐标（(u,v) 换回 (x,y)）
        t.pk = sortX ? Point2d(prof[iMax].x, prof[iMax].y)
                     : Point2d(prof[iMax].y, prof[iMax].x);
        t.tr = sortX ? Point2d(prof[iMin].x, prof[iMin].y)
                     : Point2d(prof[iMin].y, prof[iMin].x);

        // ---------- 6. 判定运动类型 ----------
        if (t.lineRMSE < 3.0) {
            t.type = "直线运动";
            t.typeEn = "line";
            t.detail = "一阶拟合残差只有 " + to_string((int)lround(t.lineRMSE)) + " px";
        } else if (cross >= 4 && t.paraRMSE > 0.6 * t.lineRMSE) {
            t.type = "正弦曲线运动";
            t.typeEn = "sine";
            t.detail = "去直线趋势后 " + to_string(cross) + " 次穿过零线，振幅约 " +
                       to_string((int)lround(t.amp)) + " px，周期约 " +
                       to_string((int)lround(t.period)) + " px";
        } else if (t.paraRMSE < 0.5 * t.lineRMSE) {
            t.type = "抛物线运动";
            t.typeEn = "parabola";
            t.detail = "二次拟合残差 " + to_string((int)lround(t.paraRMSE)) +
                       " px，远小于直线拟合的 " + to_string((int)lround(t.lineRMSE)) + " px";
        } else {
            t.type = "一般曲线运动";
            t.typeEn = "curve";
            t.detail = "直线拟合残差 " + to_string((int)lround(t.lineRMSE)) + " px";
        }

        t.start = sortX ? Point2d(prof.front().x, prof.front().y)
                        : Point2d(prof.front().y, prof.front().x);
        t.finish = sortX ? Point2d(prof.back().x, prof.back().y)
                         : Point2d(prof.back().y, prof.back().x);

        // ---------- 6.5 按图上的小字判断哪头是起点 ----------
        // 同色的小连通域聚成文字组，离哪个端点近就属于哪个端点。
        vector<TextGroup> groups = findTextGroups(m, lbl, stats, n);
        TextGroup* gStart = nullptr;
        TextGroup* gEnd = nullptr;
        double dS = 1e9, dE = 1e9;
        for (auto& g : groups) {
            double a = norm(Point2d(g.c) - t.start);
            double b = norm(Point2d(g.c) - t.finish);
            if (a < b && a < 90) { if (a < dS) { dS = a; gStart = &g; } }
            if (b <= a && b < 90) { if (b < dE) { dE = b; gEnd = &g; } }
        }
        if (gStart && gEnd) {
            // 两组都在：宽的那个是 "start"
            if (gEnd->w > gStart->w) {
                swap(t.start, t.finish);
                t.startFromLabel = true;
            } else {
                t.startFromLabel = true;
            }
        } else if (gStart) {
            // 只有一组：这张图的规矩是只给起点写字（A、B 就是这样）
            t.startFromLabel = true;
        } else if (gEnd) {
            // 只给终点写了字，说明顺序反了
            swap(t.start, t.finish);
            t.startFromLabel = true;
        }

        tracks.push_back(t);
    }

    if (tracks.empty()) {
        cerr << "没有检测到任何轨迹" << endl;
        return 2;
    }

    // ---------- 7. 画图：左边原图 + 右边数据面板 ----------
    const int PANEL = 300;
    Mat out(img.rows, img.cols + PANEL, CV_8UC3, Scalar(255, 255, 255));
    img.copyTo(out(Rect(0, 0, img.cols, img.rows)));

    // 7.1 原图上只画：端点标记、正弦的振幅/周期
    // 端点标记要小：原图的 "C--end" 这类小字就写在端点旁边，圈画大了会
    // 直接把字母盖掉。半径 3/5 只压在端点上，具体坐标另在右侧面板列出。
    for (auto& t : tracks) {
        circle(out, Point((int)t.start.x, (int)t.start.y), 3, t.color, -1);
        circle(out, Point((int)t.finish.x, (int)t.finish.y), 5, t.color, 1);

        if (t.type.find("正弦") != string::npos) {
            double ypk = t.pk.y, ytr = t.tr.y;
            int x0 = (int)min(t.pk.x, t.tr.x), x1 = (int)max(t.pk.x, t.tr.x);
            // 峰、谷两条水平虚线
            for (int x = x0; x <= x1; x += 8) {
                line(out, Point(x, (int)ypk), Point(min(x + 4, x1), (int)ypk),
                     Scalar(120, 180, 120), 1, LINE_AA);
                line(out, Point(x, (int)ytr), Point(min(x + 4, x1), (int)ytr),
                     Scalar(120, 180, 120), 1, LINE_AA);
            }
            // 峰到谷的竖直双箭头，标出落差
            int xa = x1 + 12;
            arrowedLine(out, Point(xa, (int)ypk), Point(xa, (int)ytr),
                        Scalar(0, 120, 0), 1, LINE_AA, 0, 0.04);
            arrowedLine(out, Point(xa, (int)ytr), Point(xa, (int)ypk),
                        Scalar(0, 120, 0), 1, LINE_AA, 0, 0.04);
            char buf[64];
            snprintf(buf, sizeof(buf), "%.0f px", t.resMax - t.resMin);
            putLabel(out, buf, Point(xa + 5, (int)((ypk + ytr) / 2)), Scalar(0, 120, 0));

            // 周期写在曲线下方空白处，别压在线上
            snprintf(buf, sizeof(buf), "T = %.0f px", t.period);
            putLabel(out, buf, Point(x0 + 10, (int)ytr + 22), Scalar(0, 120, 0));
        }
    }

    // 7.2 面板
    int x0 = img.cols + 14, y = 34;
    auto ptext = [&](const string& s, Scalar col, double sc, int dy) {
        putText(out, s, Point(x0, y), FONT_HERSHEY_SIMPLEX, sc, Scalar(0, 0, 0), 1, LINE_AA);
        (void)col;
        y += dy;
    };
    putText(out, "TRACK SUMMARY", Point(x0, y), FONT_HERSHEY_SIMPLEX, 0.62,
            Scalar(0, 0, 0), 2, LINE_AA);
    y += 12;
    line(out, Point(x0, y), Point(img.cols + PANEL - 14, y), Scalar(0, 0, 0), 1);
    y += 24;

    char buf[256];
    snprintf(buf, sizeof(buf), "Objects: %d", (int)tracks.size());
    ptext(buf, Scalar(0, 0, 0), 0.45, 18);

    for (auto& t : tracks) {
        rectangle(out, Rect(x0, y - 12, 12, 12), t.color, -1);
        snprintf(buf, sizeof(buf), "%s   %s", t.name.c_str(), t.typeEn.c_str());
        putText(out, buf, Point(x0 + 18, y), FONT_HERSHEY_SIMPLEX, 0.5,
                Scalar(0, 0, 0), 1, LINE_AA);
        y += 20;

        snprintf(buf, sizeof(buf), "length   %d px", (int)lround(t.length));
        ptext(buf, Scalar(0, 0, 0), 0.42, 17);
        snprintf(buf, sizeof(buf), "span     %d x %d px",
                 (int)lround(t.hspan), (int)lround(t.vspan));
        ptext(buf, Scalar(0, 0, 0), 0.42, 17);
        snprintf(buf, sizeof(buf), "start (%d,%d) end (%d,%d)",
                 (int)lround(t.start.x), (int)lround(t.start.y),
                 (int)lround(t.finish.x), (int)lround(t.finish.y));
        ptext(buf, Scalar(0, 0, 0), 0.42, 17);
        snprintf(buf, sizeof(buf), "fit RMSE line %d / quad %d",
                 (int)lround(t.lineRMSE), (int)lround(t.paraRMSE));
        ptext(buf, Scalar(0, 0, 0), 0.42, 17);
        if (t.type.find("正弦") != string::npos) {
            snprintf(buf, sizeof(buf), "amplitude %d px (peak-trough %d)",
                     (int)lround(t.amp), (int)lround(t.resMax - t.resMin));
            ptext(buf, Scalar(0, 0, 0), 0.42, 17);
            snprintf(buf, sizeof(buf), "period    %d px   peaks %d",
                     (int)lround(t.period), t.peaks);
            ptext(buf, Scalar(0, 0, 0), 0.42, 17);
        }
        y += 8;
    }

    y += 6;
    line(out, Point(x0, y), Point(img.cols + PANEL - 14, y), Scalar(180, 180, 180), 1);
    y += 18;
    putText(out, "filled dot = start", Point(x0, y), FONT_HERSHEY_SIMPLEX, 0.38,
            Scalar(90, 90, 90), 1, LINE_AA);
    y += 15;
    putText(out, "hollow ring = end", Point(x0, y), FONT_HERSHEY_SIMPLEX, 0.38,
            Scalar(90, 90, 90), 1, LINE_AA);
    y += 15;
    putText(out, "length = 4 px binned polyline", Point(x0, y),
            FONT_HERSHEY_SIMPLEX, 0.35, Scalar(90, 90, 90), 1, LINE_AA);

    // ---------- 8. 控制台报告 ----------
    cout << "检测到轨迹数量: " << tracks.size() << endl;
    for (auto& t : tracks) {
        cout << "\n" << t.cn << " 轨迹" << endl;
        cout << "  起点 (" << (int)t.start.x << ", " << (int)t.start.y << ")"
             << "  终点 (" << (int)t.finish.x << ", " << (int)t.finish.y << ")"
             << "（起点按图上的 start 标签定向）" << endl;
        cout << "  运动类型: " << t.type << "（" << t.detail << "）" << endl;
        cout << "  轨迹长度估计: " << (int)lround(t.length)
             << " px（4 px 分箱后的折线长度）" << endl;
        cout << "  跨度: 水平 " << (int)lround(t.hspan)
             << " px，竖直 " << (int)lround(t.vspan) << " px" << endl;
        if (t.type.find("正弦") != string::npos)
            cout << "  波峰 " << t.peaks << " 个，峰谷落差约 "
                 << (int)lround(t.resMax - t.resMin) << " px" << endl;
    }

    vector<int> idx(tracks.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
    sort(idx.begin(), idx.end(),
         [&](int a, int b) { return tracks[a].length > tracks[b].length; });

    cout << "\n轨迹长度排序: ";
    for (size_t i = 0; i < idx.size(); ++i) {
        cout << (i ? " > " : "") << tracks[idx[i]].cn << "("
             << (int)lround(tracks[idx[i]].length) << ")";
    }
    cout << endl;

    const Track& top = tracks[idx[0]];
    cout << "轨迹最长的物体: " << top.cn << "，" << (int)lround(top.length) << " px" << endl;
    if (idx.size() > 1) {
        const Track& second = tracks[idx[1]];
        double diff = top.length - second.length;
        if (diff < 0.1 * top.length)
            cout << "  （与第二长 " << second.cn << " 只差 "
                 << (int)lround(diff) << " px，两者几乎等长）" << endl;
    }
    cout << "竖直落差最大的是 ";
    int iv = 0;
    for (size_t i = 1; i < tracks.size(); ++i)
        if (tracks[i].vspan > tracks[iv].vspan) iv = (int)i;
    cout << tracks[iv].cn << "（" << (int)lround(tracks[iv].vspan) << " px）" << endl;

    string outPath = (argc > 2) ? argv[2] : "q10_result.png";
    imwrite(outPath, out);
    cout << "结果图已保存: " << outPath << endl;
    return 0;
}
