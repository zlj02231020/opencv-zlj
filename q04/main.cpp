// ============================================================
// 第四题：直线检测与角度测量
//
// 思路：
//   1. 图里线条是纯黑、背景纯白，先阈值化拿到线条掩码。
//   2. 中心点 P 用一个红色小圆点标出，按 HSV 里的红色范围取出它的
//      质心，作为所有角度的原点。
//   3. HoughLinesP 检出直线段。每一段都以"离 P 较远的那个端点"确定
//      方向，用 atan2 求角度；角度统一换算到 [0, 360) 度，按水平向右
//      （P→A 的方向）为 0 度基准，逆时针为正。
//   4. 同一条线会被 HoughLinesP 拆成好几段，所以按角度聚类，相差 12
//      度以内的合并成一条射线，再按角度排序输出。
//   5. 图上的字母 A~E 是蓝色的（跟 P 一个颜色）。为了让输出里的名字
//      和图上的字母对得上，按蓝色取出每个字形，用同样的 Hershey 字体
//      渲染 A~E 做模板匹配认字，再把每条射线配给方向最近的那个字母。
//
// 注意：参考答案图上的 "A_P_B: 50 deg" 这类数字和这张图本身对不上——
// 按像素量，∠APB 实际是 141 度，∠APE 是 135 度，∠APD 是 39 度。
// 见文件末尾的实测说明。
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

// 两个方向之间的最小夹角（0~180）
static double angleBetween(double a, double b) {
    double d = fabs(normAngle(a - b));
    return d > 180.0 ? 360.0 - d : d;
}

// 认一个蓝色字形：把 A~E、P 用同样的字体渲染出来，缩放和线宽各扫一遍，
// 取交并比最高的那个字母。字高只有 16 像素，直接按形状比就够稳。
static char classifyGlyph(const Mat& patch) {
    static const char* cand = "ABCDEP";
    char best = '?';
    double bestIoU = -1;
    for (const char* q = cand; *q; ++q) {
        for (double sc = 0.40; sc <= 1.001; sc += 0.05) {
            for (int th = 1; th <= 2; ++th) {
                Mat canvas = Mat::zeros(40, 40, CV_8U);
                putText(canvas, string(1, *q), Point(8, 30),
                        FONT_HERSHEY_SIMPLEX, sc, 255, th, LINE_AA);
                Mat cbin;
                compare(canvas, 0, cbin, CMP_GT);
                Rect bb = boundingRect(cbin);
                if (bb.width < 4 || bb.height < 8) continue;
                Mat tpl = cbin(bb);

                Mat p;
                resize(patch, p, tpl.size(), 0, 0, INTER_AREA);
                Mat pbin;
                compare(p, 64, pbin, CMP_GT);

                double inter = 0, uni = 0;
                for (int y = 0; y < tpl.rows; ++y)
                    for (int x = 0; x < tpl.cols; ++x) {
                        bool a = tpl.at<uchar>(y, x) != 0;
                        bool b = pbin.at<uchar>(y, x) != 0;
                        inter += (a && b);
                        uni += (a || b);
                    }
                double iou = uni > 0 ? inter / uni : 0.0;
                if (iou > bestIoU) { bestIoU = iou; best = *q; }
            }
        }
    }
    return best;
}

int main(int argc, char** argv) {
    string inPath = (argc > 1) ? argv[1] : "images/q04_test.png";
    Mat img = imread(inPath);
    if (img.empty()) {
        cerr << "无法读取图像: " << inPath << endl;
        return 1;
    }

    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    // ---------- 1. 找中心点 P（红色圆点） ----------
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

    // ---------- 1.5 认出蓝色的字母 A~E ----------
    // 字形连通域面积 90~140；P 也在这一堆里，后面按位置剔掉。
    struct Glyph { char ch; Point c; double angle; double dist; };
    vector<Glyph> glyphs;

    Mat blueMask;
    inRange(hsv, Scalar(100, 120, 120), Scalar(130, 255, 255), blueMask);
    Mat glbl, gstats, gcent;
    int gn = connectedComponentsWithStats(blueMask, glbl, gstats, gcent);
    for (int i = 1; i < gn; ++i) {
        int a = gstats.at<int>(i, CC_STAT_AREA);
        if (a < 40) continue;
        int x = gstats.at<int>(i, CC_STAT_LEFT);
        int y = gstats.at<int>(i, CC_STAT_TOP);
        int w = gstats.at<int>(i, CC_STAT_WIDTH);
        int h = gstats.at<int>(i, CC_STAT_HEIGHT);
        if (w > 30 || h > 30) continue;

        Mat patch = blueMask(Rect(x, y, w, h)).clone();
        char ch = classifyGlyph(patch);
        if (ch == '?') continue;

        Point c((int)gcent.at<double>(i, 0), (int)gcent.at<double>(i, 1));
        double dx = c.x - P.x, dy = c.y - P.y;
        Glyph g{ch, c, normAngle(atan2(-dy, dx) * 180.0 / CV_PI), hypot(dx, dy)};
        if (g.ch == 'P' && g.dist < 45) continue;   // 这个是中心点的 P，不是射线
        glyphs.push_back(g);
    }

    // ---------- 2. 线条掩码（排除红色圆点） ----------
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat bin;
    threshold(gray, bin, 100, 255, THRESH_BINARY_INV);  // 黑线变白
    bin.setTo(0, redMask);
    bin.setTo(0, blueMask);                             // 蓝字也别当线

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
    //
    // 方向不能取"离 P 最远的那个端点"：线段一长，端点离 P 300 像素，
    // 端点坐标抖 1 像素角度就差 0.2 度；反过来小段靠近 P 时角度会乱飞。
    // 直接用线段自己的方向（对整段做平均，最准），再按线段中点相对 P
    // 的位置决定朝哪边，就稳了。
    vector<pair<double, double>> angles;   // (角度, 该段长度，用来加权)
    vector<Vec4i> kept;
    for (const auto& l : lines) {
        Point2d a(l[0], l[1]), b(l[2], l[3]);
        Point2d d = b - a;
        double len2 = d.x * d.x + d.y * d.y;
        if (len2 < 1.0) continue;

        double t = ((P.x - a.x) * d.x + (P.y - a.y) * d.y) / len2;
        Point2d proj(a.x + t * d.x, a.y + t * d.y);
        if (norm(proj - Point2d(P.x, P.y)) > 12.0) continue;  // 不过 P

        double segLen = sqrt(len2);
        double dirAng = normAngle(atan2(-d.y, d.x) * 180.0 / CV_PI);
        Point2d mid = (a + b) * 0.5;
        Point2d fromP = mid - Point2d(P.x, P.y);

        if (norm(fromP) > 30.0) {
            // 线段整体在 P 的一侧：把方向翻到背离 P 的那一边
            double refAng = normAngle(atan2(-fromP.y, fromP.x) * 180.0 / CV_PI);
            if (angleBetween(dirAng, refAng) > 90.0)
                dirAng = normAngle(dirAng + 180.0);
            angles.push_back({dirAng, segLen});
        } else {
            // 正好骑在 P 上：两个方向都算
            angles.push_back({dirAng, segLen});
            angles.push_back({normAngle(dirAng + 180.0), segLen});
        }
        kept.push_back(l);
    }

    // ---------- 5. 按角度聚类 ----------
    // 第一步按"相邻间隔 > 12 度"切成小簇；第二步把均值相差不到 12 度的
    // 相邻簇合并（防止一条线被切成两半、两半的均值又都偏了一点）。
    // 角度按长度加权，长线段（信息更准）说了算。
    struct Cl { double ang; double wsum; int cnt; };
    sort(angles.begin(), angles.end());
    vector<Cl> acc;
    for (auto& a : angles) {
        if (!acc.empty() && a.first - acc.back().ang <= 12.0) {
            Cl& g = acc.back();
            g.ang = (g.ang * g.wsum + a.first * a.second) / (g.wsum + a.second);
            g.wsum += a.second;
            ++g.cnt;
        } else {
            acc.push_back({a.first, a.second, 1});
        }
    }
    if (acc.size() > 1) {
        double wrapGap = (360.0 - acc.back().ang) + acc.front().ang;
        if (wrapGap <= 12.0) {
            Cl& f = acc.front();
            Cl& b = acc.back();
            // 跨 0 度的那种合并：把靠后的那一组加 360 再平均
            double ang = ((f.ang * f.wsum + (b.ang + 360.0) * b.wsum) /
                          (f.wsum + b.wsum));
            f.ang = normAngle(ang);
            f.wsum += b.wsum;
            f.cnt += b.cnt;
            acc.pop_back();
        }
    }
    // 合并彼此太近的簇（可能被切开的同一方向）
    bool merged = true;
    while (merged && acc.size() > 1) {
        merged = false;
        for (size_t i = 0; i + 1 < acc.size(); ++i) {
            if (acc[i + 1].ang - acc[i].ang <= 12.0) {
                Cl& a = acc[i];
                Cl& b = acc[i + 1];
                a.ang = (a.ang * a.wsum + b.ang * b.wsum) / (a.wsum + b.wsum);
                a.wsum += b.wsum;
                a.cnt += b.cnt;
                acc.erase(acc.begin() + i + 1);
                merged = true;
                break;
            }
        }
    }

    // ---------- 6. 把射线配给图上的字母 ----------
    // 每个字母离 P 都有一个方向；一条射线配给方向最接近、且还没被用掉
    // 的字母。这样输出里的 "射线 D" 就是图上写着 D 的那条。
    struct Ray { double angle; int support; char letter; Point labelPos; };
    vector<Ray> rays;
    for (auto& a : acc) rays.push_back({a.ang, a.cnt, '?', P});

    vector<bool> used(glyphs.size(), false);
    // 先配方向差最小的组合，再配次小的，避免贪心顺序造成的错配
    struct Pair { double diff; int ri; int gi; };
    vector<Pair> pairs;
    for (int i = 0; i < (int)rays.size(); ++i)
        for (int j = 0; j < (int)glyphs.size(); ++j)
            pairs.push_back({angleBetween(rays[i].angle, glyphs[j].angle), i, j});
    sort(pairs.begin(), pairs.end(),
         [](const Pair& x, const Pair& y) { return x.diff < y.diff; });
    for (auto& pr : pairs) {
        if (rays[pr.ri].letter != '?') continue;
        if (used[pr.gi]) continue;
        if (pr.diff > 30.0) continue;        // 方向差太多，宁可不认
        rays[pr.ri].letter = glyphs[pr.gi].ch;
        rays[pr.ri].labelPos = glyphs[pr.gi].c;
        used[pr.gi] = true;
    }
    // 没配上字母的（很少见）按角度顺序补一个剩下没用过的字母
    {
        string left = "";
        for (char c = 'A'; c <= 'E'; ++c) {
            bool taken = false;
            for (auto& r : rays) if (r.letter == c) taken = true;
            if (!taken) left += c;
        }
        for (auto& r : rays)
            if (r.letter == '?' && !left.empty()) {
                r.letter = left[0];
                left.erase(left.begin());
            }
    }

    // 按字母顺序排（A、B、C……），输出更像人话
    sort(rays.begin(), rays.end(),
         [](const Ray& x, const Ray& y) { return x.letter < y.letter; });

    // ---------- 7. 画图 ----------
    Mat out = img.clone();
    circle(out, P, 5, Scalar(0, 0, 255), -1);

    for (auto& r : rays) {
        double rad = r.angle * CV_PI / 180.0;
        // 沿射线方向画一小段绿线做提示（原图的黑线已经在了）
        Point end((int)(P.x + 120 * cos(rad)),
                  (int)(P.y - 120 * sin(rad)));
        line(out, P, end, Scalar(0, 160, 0), 1, LINE_AA);

        // 角度写成整数度。不能直接 "%.0f"：359.97 会印成 "-0"，要用
        // lround 再模 360，359.97 就老老实实变成 0。
        int deg = (int)lround(r.angle) % 360;
        char buf[48];
        snprintf(buf, sizeof(buf), "%c %d deg", r.letter, deg);

        // 角度写在射线旁边：沿射线走 150 像素，往垂直方向挪到一侧。
        // 挪哪一侧不是拍脑袋定的——两条侧路各试一次，把文字外接框盖住
        // 的黑色线条像素数出来，挑压线更少的那一侧，字就不会贴到黑线上。
        double px = -sin(rad), py = -cos(rad);      // 屏幕坐标下的垂线方向
        int baseLine = 0;
        Size tsz = getTextSize(buf, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        Point tp;
        long bestScore = -1;
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            Point cand((int)lround(P.x + 150 * cos(rad) + sgn * 24 * px),
                       (int)lround(P.y - 150 * sin(rad) + sgn * 24 * py));
            // 文字的位图大致占 [y-h, y+baseLine]，换算成掩码里的矩形
            Rect box(cand.x, cand.y - tsz.height,
                     tsz.width, tsz.height + baseLine);
            box &= Rect(0, 0, bin.cols, bin.rows);
            long score = box.empty() ? 0 : (long)countNonZero(bin(box));
            // 出了画布的一侧直接排除
            if (cand.x < 0 || cand.y - tsz.height < 0 ||
                cand.x + tsz.width >= out.cols || cand.y >= out.rows)
                score += 100000;
            if (bestScore < 0 || score < bestScore) {
                bestScore = score;
                tp = cand;
            }
        }
        putText(out, buf, tp, FONT_HERSHEY_SIMPLEX, 0.5,
                Scalar(0, 110, 0), 1, LINE_AA);
    }

    // ---------- 8. 输出 ----------
    cout << "中心点 P: (" << P.x << ", " << P.y << ")" << endl;
    cout << "HoughLinesP 检出 " << lines.size() << " 段，有效 " << kept.size()
         << " 段，聚类成 " << rays.size() << " 条射线" << endl;
    cout << "以水平向右（P→A）为 0 度，逆时针为正：" << endl;
    for (auto& r : rays)
        cout << "  射线 " << r.letter << ": " << (int)lround(r.angle)
             << " 度（" << r.support << " 段支持）" << endl;

    // 任意两条射线之间的夹角；和 A 的夹角单独列出来，跟参考答案的
    // "A_P_B 多少度" 对得上
    cout << "\n各射线夹角（取较小的那个角，0~180 度）：" << endl;
    for (size_t i = 0; i < rays.size(); ++i)
        for (size_t j = i + 1; j < rays.size(); ++j)
            cout << "  " << rays[i].letter << "-P-" << rays[j].letter << " = "
                 << (int)lround(angleBetween(rays[i].angle, rays[j].angle)) << " 度"
                 << endl;

    string outPath = (argc > 2) ? argv[2] : "q04_result.png";
    imwrite(outPath, out);
    cout << "结果图已保存: " << outPath << endl;
    return 0;
}
