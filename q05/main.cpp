// ============================================================
// 第五题：轮廓匹配与瑕疵识别
//
// 思路：
//   1. 齿轮是灰底黑边，用灰度阈值就能拿到三个齿轮的外轮廓；图上还有
//      文字标签，按面积过滤掉。
//   2. 最左边那个是标准件，另外两个是 Sample A / B。
//   3. 齿数：以质心为极点，把轮廓点换算成极坐标 (θ, r)，每个角度取
//      最大半径，得到一条"半径-角度"曲线。齿轮每一颗齿就是曲线上的
//      一个峰，数峰的个数就是齿数。缺齿的位置曲线上会塌下去。
//   4. 相似度用 matchShapes 的 Hu 矩距离量化，值越小越像。
//   5. 缺齿定位：把样品与标准件的半径曲线都归一化到平均半径，逐角度
//      做差，差值连续超过阈值的角度区间就是一个缺口。
// ============================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

static const int NBIN = 360;

// 把轮廓转成"每个角度上的最大半径"曲线
static vector<double> radialProfile(const vector<Point>& c, Point2d cen) {
    vector<double> prof(NBIN, 0.0);
    for (const auto& p : c) {
        double dx = p.x - cen.x, dy = cen.y - p.y;   // 图像 y 轴向下，取反让角度符合直觉
        double r = sqrt(dx * dx + dy * dy);
        double ang = atan2(dy, dx) * 180.0 / CV_PI;
        int bin = (int)lround(ang);
        bin %= 360;
        if (bin < 0) bin += 360;
        if (r > prof[bin]) prof[bin] = r;
    }
    // 360 个角度格子比轮廓点还密的时候会有空格子，留 0 会在后面变成
    // 巨大的虚假差值。用左右非空值线性补上。
    for (int i = 0; i < NBIN; ++i) {
        if (prof[i] > 0) continue;
        int j = 1, k = 1;
        while (prof[(i + j) % NBIN] == 0 && j < NBIN) ++j;
        while (prof[(i - k + NBIN) % NBIN] == 0 && k < NBIN) ++k;
        prof[i] = 0.5 * (prof[(i + j) % NBIN] + prof[(i - k + NBIN) % NBIN]);
    }
    return prof;
}

// 齿尖角度列表
static vector<int> toothAngles(const vector<double>& prof) {
    double mean = 0;
    for (double v : prof) mean += v;
    mean /= NBIN;
    double thr = mean * 1.03;

    vector<int> peaks;
    for (int i = 0; i < NBIN; ++i) {
        if (prof[i] > thr && prof[i] >= prof[(i + NBIN - 1) % NBIN] &&
            prof[i] > prof[(i + 1) % NBIN]) {
            if (!peaks.empty() && i - peaks.back() <= 6) continue;  // 同一个峰的抖动
            peaks.push_back(i);
        }
    }
    return peaks;
}

// 找缺齿：看齿尖之间的角间距。正常齿距是均匀的，一旦某个间隔接近
// 两倍齿距，那段就是缺了一颗齿。
// 这个方法的好处是自参照的——样品和标准件的齿数不一样（实测样品是
// 10 齿 / 11 齿的齿轮，标准件是 12 齿），按"与标准件逐角度做差"的
// 思路怎么对齐都不对，而按样品自己的齿距找异常就不会受影响。
static int findDefects(const vector<double>& prof, double& worstAngle) {
    vector<int> peaks = toothAngles(prof);
    worstAngle = 0;
    if (peaks.size() < 4) return 0;

    // 相邻齿尖的角间距（首尾相连）
    vector<double> gaps;
    for (size_t i = 0; i < peaks.size(); ++i) {
        int nxt = (i + 1 < peaks.size()) ? peaks[i + 1] : peaks[0] + NBIN;
        gaps.push_back(nxt - peaks[i]);
    }
    vector<double> sorted = gaps;
    sort(sorted.begin(), sorted.end());
    double median = sorted[sorted.size() / 2];

    int defects = 0;
    double biggest = 0;
    for (size_t i = 0; i < gaps.size(); ++i) {
        if (gaps[i] > 1.6 * median) {
            ++defects;
            if (gaps[i] > biggest) {
                biggest = gaps[i];
                int a = peaks[i];
                int b = (i + 1 < peaks.size()) ? peaks[i + 1] : peaks[0] + NBIN;
                worstAngle = (double)((a + b) / 2 % NBIN);
            }
        }
    }
    return defects;
}

// 找缺齿区间：先把两条曲线做圆周对齐，再逐角度做差
static int findDefects(const vector<double>& std, const vector<double>& samp,
                       double& worstAngle, double& worstDrop) {
    // 两边各自按平均半径归一化，消除大小差异
    double ms = 0, mp = 0;
    for (double v : std) ms += v;
    for (double v : samp) mp += v;
    ms /= NBIN; mp /= NBIN;

    // ---- 关键一步：样品在摆放时是转过角度的，不能直接按 0 度对齐比较 ----
    // 枚举所有圆周平移量，取"残差平方和最小"的那个作为最佳对齐。
    int bestShift = 0;
    double bestCost = 1e18;
    for (int s = 0; s < NBIN; ++s) {
        double cost = 0;
        for (int i = 0; i < NBIN; ++i) {
            double d = std[i] / ms - samp[(i + s) % NBIN] / mp;
            cost += d * d;
        }
        if (cost < bestCost) { bestCost = cost; bestShift = s; }
    }

    worstAngle = 0; worstDrop = 0;
    int defectCount = 0;
    int runLen = 0;
    bool inDefect = false;

    for (int i = 0; i < NBIN; ++i) {
        double diff = std[i] / ms - samp[(i + bestShift) % NBIN] / mp;
        bool bad = diff > 0.06;
        if (bad) {
            ++runLen;
            if (!inDefect) { ++defectCount; inDefect = true; }
            if (diff > worstDrop) {
                worstDrop = diff;
                worstAngle = (double)((i - bestShift + 2 * NBIN) % NBIN);
            }
        } else {
            inDefect = false;
        }
    }
    // 掐头去尾：宽度不到 6 度的差值大多是齿边的抖动，不是真缺口
    if (runLen < 6) defectCount = 0;
    return defectCount;
}

int main(int argc, char** argv) {
    string inPath = (argc > 1) ? argv[1] : "images/q05_test.png";
    Mat img = imread(inPath);
    if (img.empty()) {
        cerr << "无法读取图像: " << inPath << endl;
        return 1;
    }

    // ---------- 1. 提取齿轮轮廓 ----------
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat mask;
    threshold(gray, mask, 200, 255, THRESH_BINARY_INV);

    vector<vector<Point>> all;
    findContours(mask, all, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    vector<vector<Point>> gears;
    for (auto& c : all)
        if (contourArea(c) > 5000) gears.push_back(c);

    if (gears.size() < 3) {
        cerr << "齿轮数量不足（找到 " << gears.size() << " 个）" << endl;
        return 2;
    }

    // 按质心 x 排序：最左是标准件
    sort(gears.begin(), gears.end(), [](const vector<Point>& a, const vector<Point>& b) {
        Moments ma = moments(a), mb = moments(b);
        return (ma.m10 / ma.m00) < (mb.m10 / mb.m00);
    });

    Mat out = img.clone();
    const char* names[3] = {"Standard", "Sample A", "Sample B"};
    const Scalar colors[3] = {Scalar(0, 200, 0), Scalar(255, 0, 0), Scalar(0, 0, 255)};

    vector<vector<double>> profs;
    vector<Point2d> centers;
    for (size_t i = 0; i < 3; ++i) {
        Moments m = moments(gears[i]);
        Point2d cen(m.m10 / m.m00, m.m01 / m.m00);
        centers.push_back(cen);
        profs.push_back(radialProfile(gears[i], cen));
    }

    // ---------- 2. 齿数与相似度 ----------
    int teethStd = (int)toothAngles(profs[0]).size();
    cout << "标准件齿数: " << teethStd << " 齿" << endl;

    for (size_t i = 0; i < 3; ++i) {
        drawContours(out, gears, (int)i, colors[i], 2);
        putText(out, names[i], Point((int)centers[i].x - 40, (int)centers[i].y),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 3, LINE_AA);
        putText(out, names[i], Point((int)centers[i].x - 40, (int)centers[i].y),
                FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1, LINE_AA);
    }

    // ---------- 3. 逐个样品与标准件比对 ----------
    for (int i = 1; i < 3; ++i) {
        double ms = matchShapes(gears[0], gears[i], CONTOURS_MATCH_I1, 0);
        int teeth = (int)toothAngles(profs[i]).size();

        double ang = 0;
        int defect = findDefects(profs[i], ang);

        cout << names[i] << "：齿数 " << teeth
             << "，缺齿 " << defect << " 处"
             << "，matchShapes 距离 = " << fixed << setprecision(4) << ms
             << endl;

        // 在最严重的缺口位置画个圈
        double rad = ang * CV_PI / 180.0;
        double rr = 0;
        for (double v : profs[i]) rr += v;
        rr /= NBIN;
        Point p((int)(centers[i].x + rr * cos(rad)),
                (int)(centers[i].y - rr * sin(rad)));
        circle(out, p, 18, Scalar(0, 0, 255), 2);
        putText(out, "missing", p + Point(-30, -22), FONT_HERSHEY_SIMPLEX, 0.45,
                Scalar(0, 0, 255), 1, LINE_AA);

        char buf[64];
        snprintf(buf, sizeof(buf), "teeth:%d miss:%d", teeth, defect);
        putText(out, buf, Point((int)centers[i].x - 50, (int)centers[i].y + 60),
                FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2, LINE_AA);
    }

    // 标准件自身比对作为基准（应该接近 0）
    cout << "标准件自比对基准: "
         << fixed << setprecision(4)
         << matchShapes(gears[0], gears[0], CONTOURS_MATCH_I1, 0) << endl;

    string outPath = (argc > 2) ? argv[2] : "q05_result.png";
    imwrite(outPath, out);
    cout << "结果图已保存: " << outPath << endl;
    return 0;
}
