import numpy
import cv2
import math
# ----------------------------------------------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from skimage import exposure
from skimage  import  transform as frnsf
from scipy import signal
# ----------------------------------------------------------------------------------------------------------------------
from function.soccer_utils.tools import tools_IO
from function.soccer_utils.tools import tools_Skeletone
from function.soccer_utils.tools import tools_image
from function.soccer_utils.tools import tools_draw_numpy
from function.soccer_utils.tools import tools_render_CV
from function.soccer_utils.tools import tools_pr_geom
# ----------------------------------------------------------------------------------------------------------------------
from function.soccer_utils.tools import utils_draw
from function.soccer_utils.tools import utils_homography
from function.soccer_utils.tools import utils_GT_data
from function.soccer_utils.tools import utils_Time_Profiler
# ----------------------------------------------------------------------------------------------------------------------
class Soccer_Field_Lines_Extractor(object):
    def __init__(self):
        # self.folder_out = folder_out

        self.HE = utils_homography.Homography_Engine()
        self.GT_data = utils_GT_data.Soccer_Field_GT_data()
        self.Ske = tools_Skeletone.Skelenonizer()
        self.T = utils_Time_Profiler.Time_Profiler()

        self.W = None
        self.H = None

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_angle(self, line):
        x1, y1, x2, y2 = line
        if x2 - x1==0:
            angle = 0
        else:
            angle = 90 + math.atan((y2 - y1) / (x2 - x1)) * 180 / math.pi
        return angle
# ----------------------------------------------------------------------------------------------------------------------
    def hitmap_to_segments(self,hitmap,do_skeletonize=False,max_count=5,min_q=150):
        th = numpy.quantile(hitmap.ravel(), 0.98)
        value, image_bin = cv2.threshold(hitmap, th, 255, cv2.THRESH_BINARY)

        if do_skeletonize:
            image_bin = self.Ske.binarized_to_skeleton(image_bin)

        max_label, labels, stats, centroids = cv2.connectedComponentsWithStats(image_bin.astype(numpy.uint8))

        lengths = numpy.array([numpy.linalg.norm((h, w)) for h,w in stats[1:,2:4]])
        idx = numpy.argsort(-lengths)[:max_count]

        segments = []
        for i in idx:
            segment = numpy.array(numpy.where(labels == 1+i)).T
            segment = segment[:, [1, 0]]
            q = numpy.average(hitmap[numpy.where(labels == 1+i)])
            if q>=min_q:
                segments.append(segment)

        segments = numpy.array(segments)

        return segments
# ----------------------------------------------------------------------------------------------------------------------
    def estimate_hit(self, line_cand_boxed,lines):

        tol = 5
        D = numpy.array([tools_render_CV.distance_extract_to_line(line, line_cand_boxed) for line in lines])
        is_match = numpy.array(D<=tol)
        idx_match = numpy.where(is_match)[0]

        return idx_match
# ----------------------------------------------------------------------------------------------------------------------
    def refine_line(self, segments, idx):
        if len(idx) == 0: return None
        X, Y = [], []
        for i in idx:
            X += segments[i][:, 0].tolist()
            Y += segments[i][:, 1].tolist()

        seg = numpy.vstack((X, Y), ).T
        line = self.Ske.interpolate_segment_by_line(seg)
        return line
# ----------------------------------------------------------------------------------------------------------------
    def get_best_join(self,hitmap,segments,lines,do_debug=False):
        len_segments = len(segments)

        processed = numpy.zeros((len_segments, len_segments))

        weights = numpy.array([self.Ske.get_length_segment(s) for s in segments])

        Q, Cands = {}, {}
        for s1 in range(len_segments):
            for s2 in range(s1, len_segments):
                if processed[s1, s2] == 1: continue
                processed[s1, s2] = 1

                line_cand = self.Ske.interpolate_segment_by_line(numpy.vstack((segments[s1], segments[s2])))
                line_cand_boxed = self.HE.boxify_lines([line_cand], (0, 0, hitmap.shape[1], hitmap.shape[0]))[0]

                if len(line_cand_boxed)==0:continue

                idx_match = self.estimate_hit(line_cand_boxed, lines)
                if (s1 not in idx_match) or (s2 not in idx_match):idx_match = []

                for i1 in idx_match:
                    for i2 in idx_match:
                        processed[i1, i2] = 1

                Q[(s1, s2)] = weights[idx_match].sum()
                Cands[(s1, s2)] = idx_match

        line, quality,cands = None, 0,[]
        if len(Q) > 0:
            key = tools_IO.max_element_by_value(Q)[0]
            cands = Cands[key]
            line = self.refine_line(segments, cands)

        # if do_debug:
        #     image_debug = numpy.full((self.H,self.W,3),64,dtype=numpy.uint8)
        #     image_debug = tools_draw_numpy.draw_lines(image_debug, lines, color=utils_draw.color_amber, w=1)
        #     image_debug = tools_draw_numpy.draw_lines(image_debug, numpy.array(lines)[cands], color=utils_draw.color_red, w=1)
        #     cv2.imwrite(self.folder_out + 'lines_ransac.png',image_debug)

        return line
# ----------------------------------------------------------------------------------------------------------------------
    def hitmap_to_line(self,hitmap,min_len=100,min_q=200,do_ransac=True,base_name=None,do_debug=False):

        segments = self.hitmap_to_segments(hitmap,do_skeletonize=True,max_count=5,min_q=min_q)
        segments_straight = []
        if not do_ransac:
            line = self.Ske.interpolate_segment_by_line(segments[0])
        else:
            segments_straight = self.Ske.sraighten_segments(segments, min_len)
            lines = [self.Ske.interpolate_segment_by_line(segment) for segment in segments_straight]
            line = self.get_best_join(hitmap,segments_straight,lines,do_debug=do_debug)

        # if do_debug:
        #     cv2.imwrite(self.folder_out + base_name + '_segments.png', tools_draw_numpy.draw_segments(tools_image.saturate(hitmap), segments,color=tools_draw_numpy.get_colors(len(segments),shuffle=True),w=1,put_text=True))
        #     if len(segments_straight)>0:
        #         cv2.imwrite(self.folder_out + base_name + '_segments_str.png',tools_draw_numpy.draw_segments(tools_image.saturate(hitmap), segments_straight, color=tools_draw_numpy.get_colors(len(segments_straight),shuffle=True), w=1))

        return line
# ----------------------------------------------------------------------------------------------------------------------
    def filter_hor(self, gray2d, sobel_H=9, sobel_W = 9, skip_agg=False):

        sobel = numpy.full((sobel_H, sobel_W),+1, dtype=numpy.float32)
        sobel[:,  sobel.shape[1] // 2:] = +1
        sobel[:, :sobel.shape[1] // 2 ] = -1
        if sobel.sum() > 0:
            sobel = sobel / sobel.sum()
        filtered = cv2.filter2D(gray2d, 0, sobel)

        if skip_agg:
            return filtered
        else:
            agg = tools_image.sliding_2d(filtered, -sobel_H, +sobel_H, -(sobel_W//4),+(sobel_W//4))
            neg = numpy.roll(agg,-sobel_W//4, axis=1)
            pos = numpy.roll(agg,+sobel_W//4, axis=1)
            hit = ((255-neg)+pos)/2
            hit[:,  :3] = 0
            hit[:,-3: ] = 0

        return hit
# ----------------------------------------------------------------------------------------------------------------------
    def filter_ver(self, gray2d, sobel_H, sobel_W,skip_agg=False):
        sobel = numpy.full((sobel_H, sobel_W), +1, dtype=numpy.float32)
        sobel[ sobel.shape[0] // 2:, :] = +1
        sobel[:sobel.shape[0] // 2,  :] = -1
        if sobel.sum()>0:
            sobel = sobel / sobel.sum()
        filtered = cv2.filter2D(gray2d, 0, sobel)

        if skip_agg:
            return filtered

        agg = tools_image.sliding_2d(filtered, -(sobel_H // 4), +(sobel_H // 4), -sobel_W, +sobel_W)
        neg = numpy.roll(agg, -sobel_H // 4, axis=0)
        pos = numpy.roll(agg, +sobel_H // 4, axis=0)
        hit = ((255 - neg) + pos) / 2
        hit[  :sobel_H,:] = 128
        hit[-sobel_H:, :] = 128
        return numpy.array(hit,dtype=numpy.uint8)
# ----------------------------------------------------------------------------------------------------------------------
    def get_upper_bound(self, image, grass_mask, base_name=None, do_debug=False):

        # X, success = tools_IO.load_if_exists(self.folder_out + 'cache/' + base_name + '_line_up.dat')
        # if (not do_debug) and success:
        #     return X

        hit = 255 - self.filter_ver(255 * grass_mask, 4, 13)
        hit[int(0.5 * image.shape[0]):, :] = 0
        line = self.hitmap_to_line(hit, min_len=100, do_ransac=True, base_name=base_name, do_debug=do_debug)

        if (line is not None) and numpy.linalg.norm(line[:2] - line[2:]) < image.shape[1] / 4:
            line = None

        if line is not None:
            line = self.HE.boxify_lines([line], (0, 0, image.shape[1], image.shape[0]))[0]

        # tools_IO.write_cache(self.folder_out + 'cache/' + base_name + '_line_up.dat', line)
        #
        # if do_debug:
        #     # cv2.imwrite(self.folder_out+'upper_filter.png',hit)
        #     image_debug = tools_image.desaturate(64 * grass_mask)
        #     image_debug = tools_draw_numpy.draw_lines(image_debug, [line], color=(0, 128, 255), w=1)
        #     cv2.imwrite(self.folder_out + base_name + '_bounds.png', image_debug)

        return line
# ----------------------------------------------------------------------------------------------------------------------
    def get_line_midfield(self,gray,L,base_name=None, do_debug=False):

        # X, success = tools_IO.load_if_exists(self.folder_out + 'cache/' + base_name + '_line_midfield.dat')
        # if (not do_debug) and success:
        #     return X

        hit = self.filter_hor(gray[:,:,0],sobel_H=5,sobel_W = 15)
        line = self.hitmap_to_line(hit,min_len=50,do_ransac=True,base_name=base_name, do_debug=do_debug)

        if line is not None and numpy.linalg.norm(line[:2]-line[2:])<gray.shape[0]/4:
            line = None

        L.line_midfield = line
        L.cut_miffield()

        # tools_IO.write_cache(self.folder_out + 'cache/' + base_name + '_line_midfield.dat', line)
        #
        # if do_debug:
        #     cv2.imwrite(self.folder_out + base_name + 'midline_hit.png', hit)

        return line
# ----------------------------------------------------------------------------------------------------------------------
    def get_lower_bound(self,gray3d, base_name=None, do_debug=False):

        # X, success = tools_IO.load_if_exists(self.folder_out + 'cache/' + base_name + '_line_down.dat')
        # if (not do_debug) and success:
        #     return X

        hit = self.filter_ver(gray3d[:,:,0],11,19)
        hit[:int(0.5*gray3d.shape[0]),:]=0
        line = self.hitmap_to_line(hit,min_len=200,do_ransac=True,base_name=base_name,do_debug=do_debug)

        if (line is not None) and numpy.linalg.norm(line[:2]-line[2:])<gray3d.shape[1]/3:
            line = None

        # tools_IO.write_cache(self.folder_out + 'cache/' + base_name + '_line_down.dat', line)
        #
        # if do_debug:
        #     cv2.imwrite(self.folder_out + base_name + 'hit.png', hit)
        #     cv2.imwrite(self.folder_out + base_name + 'ske.png', tools_draw_numpy.draw_lines(gray3d, [line], (0, 0, 255), w=1))

        return line
# ----------------------------------------------------------------------------------------------------------------------
    def get_line_side_L(self, gray, base_name=None, do_debug=False):

        # X, success = tools_IO.load_if_exists(self.folder_out + 'cache/' + base_name + '_line_side_L.dat')
        # if (not do_debug) and success:
        #     return X

        gray2 = tools_image.skew_hor(gray, +gray.shape[1])
        hit2 = self.filter_hor(gray2[:, :, 0], sobel_H=5, sobel_W=15, skip_agg=False)
        hit = tools_image.skew_hor(hit2, +gray.shape[1], do_inverce=True)
        line = self.hitmap_to_line(hit, do_ransac=True, min_q=150, base_name=base_name, do_debug=do_debug)

        if line is not None and numpy.linalg.norm(line[:2] - line[2:]) < gray.shape[0] / 4:
            line = None

        # tools_IO.write_cache(self.folder_out + 'cache/' + base_name + '_line_side_L.dat', line)
        #
        # if do_debug:
        #     cv2.imwrite(self.folder_out + base_name + 'hit.png', hit)

        return line
# ----------------------------------------------------------------------------------------------------------------------
    def get_line_side_R(self, gray, base_name=None, do_debug=False):

        # X, success = tools_IO.load_if_exists(self.folder_out + 'cache/' + base_name + '_line_side_R.dat')
        # if (not do_debug) and success:
        #     return X

        gray2 = tools_image.skew_hor(gray,-gray.shape[1])
        hit2 = self.filter_hor(gray2[:, :, 0],sobel_H=5, sobel_W = 15,skip_agg=False)
        hit = tools_image.skew_hor(hit2, -gray.shape[1],do_inverce=True)
        line = self.hitmap_to_line(hit, do_ransac=True, min_q=150,base_name=base_name, do_debug=do_debug)

        if line is not None and numpy.linalg.norm(line[:2]-line[2:])<gray.shape[0]/4:
            line = None

        # tools_IO.write_cache(self.folder_out + 'cache/' + base_name + '_line_side_R.dat', line)
        #
        # if do_debug:
        #     cv2.imwrite(self.folder_out + base_name + 'hit.png', hit)

        return line
# ----------------------------------------------------------------------------------------------------------------------
    def is_horizontal(self,angle):

        if angle>180:angle-=180
        if 90-10 <= angle < 90 + 10: return True
        if 0     <= angle < 90 - 10: return False
        if 90+10 <= angle < 180   : return False

        return None
# ----------------------------------------------------------------------------------------------------------------
    def prepare_vanishing_segments_convolve(self,gray,base_name,do_debug=False):

        result = self.filter_hor(gray,8,17,skip_agg=True)
        result2 = numpy.maximum(255 - result, result)
        result2 = exposure.adjust_gamma(result2, 6)

        image_bin = self.Ske.binarize(result2)
        image_ske = self.Ske.binarized_to_skeleton(image_bin)
        segemnts = self.Ske.skeleton_to_segments(image_ske)

        # if do_debug:
        #     image = tools_draw_numpy.draw_segments(tools_image.desaturate(result2),segemnts,color=(0,0,255))
        #     cv2.imwrite(self.folder_out + base_name + 'vanish_filtred.png', result)
        #     cv2.imwrite(self.folder_out + base_name + 'vanish_segm.png',image)


        return segemnts
# ----------------------------------------------------------------------------------------------------------------
    def prepare_vanishing_segments_canny(self, gray, base_name, do_debug=False):

        adjusted = exposure.adjust_sigmoid(gray, cutoff=0.5, gain=20)
        image_ske = cv2.Canny(adjusted,20, 255, apertureSize=3)
        segemnts = self.Ske.skeleton_to_segments(image_ske)
        # if do_debug:
        #     cv2.imwrite(self.folder_out + base_name + 'segm_adjusted.png', adjusted)
        #     cv2.imwrite(self.folder_out + base_name + 'segm_ske.png', image_ske)

        return segemnts
# ----------------------------------------------------------------------------------------------------------------
    def intersection_ransac(self, lines,line_ref=None):

        weights = numpy.array([numpy.linalg.norm(line[:2] - line[2:]) for line in lines])
        max_lines = 50
        idx = numpy.argsort(-weights)
        lines_best=lines[idx[:max_lines]]
        weights_best = weights[idx[:max_lines]]

        intersections = []

        len_lines_best = len(lines_best)
        
        if line_ref is not None:
            for i1 in range(len_lines_best):
                intersection = tools_render_CV.line_intersection(line_ref,lines[i1])
                if numpy.any(numpy.isnan(intersection)):continue
                if intersection[1]>self.H/2: continue
                if intersection[1]<-3*self.H : continue
                intersections.append(intersection)

        else:
            for i1 in range(len_lines_best-1):
                for i2 in range(i1+1, len_lines_best):
                    intersection = tools_render_CV.line_intersection(lines_best[i1], lines_best[i2])
                    if numpy.any(numpy.isnan(intersection)): continue

                    if intersection[1] > self.H / 2: continue
                    if intersection[1] < -3 * self.H: continue
                    intersections.append(intersection)

        if len(intersections)==0:
            return None,[],[],[]

        tol_d = 20
        Q = []
        for p in intersections:
            q = 0
            for line, weight in zip(lines_best, weights_best):
                d = tools_render_CV.distance_point_to_line(line, p)
                if d<=tol_d:
                    q += weight
            Q.append(q)

        i = int(numpy.argmax(numpy.array(Q)))
        result_point = intersections[i]

        lines_match = []
        for line in lines:
            if tools_render_CV.distance_point_to_line(line, result_point)<tol_d:
                lines_match.append(line)

        return result_point, intersections, numpy.array(lines_best),numpy.array(lines_match)
# ----------------------------------------------------------------------------------------------------------------
    def detect_vanishing_point(self,gray,line_upper_bound,line_midfield,base_name=None,do_debug=False):

        # X, success = tools_IO.load_if_exists(self.folder_out + 'cache/' + base_name + '_point_v.dat')
        # if (not do_debug) and success:
        #     return X

        if line_upper_bound is not None:
            th_row = int(5 + (line_upper_bound[1] + line_upper_bound[3]) / 2)
            ROI = gray[th_row:, :]
        else:
            th_row = 0
            ROI = gray.copy()

        segments = self.prepare_vanishing_segments_convolve(ROI,base_name)
        segments = self.Ske.sraighten_segments(segments,min_len=50)

        for s in range(len(segments)):
            for p in range(len(segments[s])):
                segments[s][p][1] += th_row

        th_ang = 75
        lines = []
        for i,segment in enumerate(segments):
            bbox = cv2.boundingRect(segment)
            if th_row is not None and bbox[1] < th_row: continue
            line = self.Ske.interpolate_segment_by_line(segment)
            angle = self.get_angle(line)

            if line_midfield is not None:
                if bbox[0]>(line_midfield[0]+line_midfield[2])/2:
                    if not ((360-th_ang<=angle<=360)or(180-th_ang<=angle<=180)):continue
                else:
                    if not ((0<=angle<=th_ang)or(180<=angle<=180+th_ang)):continue

            lines.append(line)

        if line_midfield is not None:lines.append(line_midfield)
        lines = numpy.array(lines)
        self.H, self.W = gray.shape[:2]
        result_point, intersections, lines_best, lines_match = self.intersection_ransac(lines)

        # tools_IO.write_cache(self.folder_out + 'cache/' + base_name + '_point_v.dat', result_point)
        #
        # if do_debug:
        #     cv2.imwrite(self.folder_out + base_name + '_vanishing_segm.png', tools_draw_numpy.draw_segments(tools_image.saturate(gray), segments, color=(0, 0, 255)))
        #
        #     factor = 2
        #     image= utils_draw.extend_view_from_image(gray*0.5,factor)
        #     image = tools_draw_numpy.draw_lines(image, utils_draw.extend_view(lines.reshape((-1,2)), self.H, self.W, factor=factor).reshape((-1,4)), color=(0, 128, 128), w=1)
        #
        #     if result_point is not None:
        #         image = tools_draw_numpy.draw_lines (image, utils_draw.extend_view(lines_best.reshape((-1, 2)), self.H, self.W,factor=factor).reshape((-1, 4)),color=(0, 128, 255), w=1,put_text=False)
        #         image = tools_draw_numpy.draw_lines (image, utils_draw.extend_view(lines_match.reshape((-1, 2)),self.H, self.W,factor=factor).reshape((-1, 4)),color=(0, 0, 255), w=1)
        #         image = tools_draw_numpy.draw_points(image, utils_draw.extend_view(intersections, self.H, self.W, factor=factor),color=(0, 0, 128), w=1)
        #         image = tools_draw_numpy.draw_points(image,[utils_draw.extend_view(result_point, self.H, self.W, factor=factor)],color=(0, 0, 255), w=5)
        #
        #     cv2.imwrite(self.folder_out+base_name+'_vanishing.png',image)

        return result_point
# ----------------------------------------------------------------------------------------------------------------
    def get_best_path(self, A, step,clamp_start=None,clamp_end=None):

        tol = 3
        step_min, step_max = step - tol, step + tol

        B = numpy.full(len(A), 0, dtype=numpy.int64)
        D = numpy.full(len(A), -1, dtype=numpy.int32)
        for pos in range(len(A)):
            best = None
            for c in range(pos - step_max, pos - step_min + 1):
                if c >= 0 and (best is None or B[c] > best):
                    best = B[c]
                    D[pos] = c

            if best is not None:
                B[pos] = A[pos] + best
            else:
                D[pos] = -1

        # walk back
        end = len(B) - step_max * 2 + numpy.argmax(B[len(B) - step_max * 2:])
        res = []
        pos = end
        while D[pos] > 0:
            res.append(pos)
            pos = D[pos]

        return B[end], res
# ----------------------------------------------------------------------------------------------------------------
    def get_path_back(self,end_position,Summ,Direction,State):

        zebra = numpy.full(len(Summ),0)
        keypoints = []
        pos = end_position
        while Direction[pos] >= 0:
            #print(pos,Summ[pos],A[pos-100+1:pos+1].sum())
            zebra[Direction[pos]+1:pos+1] = +State[pos]
            keypoints.append(pos)
            pos = Direction[pos]

        keypoints.append(pos)
        zebra[:pos+1] = +1*(State[pos]>0)

        return keypoints, zebra
# ----------------------------------------------------------------------------------------------------------------
    def get_best_path_zebra(self, A0, step_min, step_max, n_periods, base_name= None, debug_flip=False, do_debug=False):

        if len(A0)<step_min:
            return []

        A=A0.astype(numpy.int32)
        A-=int(A.mean())

        if A[:step_min].mean()<0:
            A=-A
        
        len_A = len(A)
        
        Q         = numpy.full(len_A,-1,dtype=numpy.float32)
        Summ      = numpy.full(len_A, 0,dtype=numpy.int64)
        Direction = numpy.full(len_A,-1,dtype=numpy.int32)
        State     = numpy.full(len_A, 0,dtype=numpy.int64) #0-invalid, 1-pos, (-1)-neg

        State[0]=+1
        Summ[0]=A[0]*State[0]
        Q[0] = 0

        for position in range(len_A):

            best_value, best_candidate = None,None
            for candidate in range(position-step_max,position-step_min+1):
                if candidate<0 or (State[candidate]==0):continue

                if State[candidate]>0:value = Summ[candidate] + numpy.sum(A[candidate+1:position+1])
                else:                 value = Summ[candidate] - numpy.sum(A[candidate+1:position+1])


                if best_value is None or value>best_value:
                    best_candidate = candidate
                    best_value = value


            if best_value is not None:
                Summ[position] = best_value
                Direction[position] = best_candidate
                State[position] = -State[best_candidate]-2*(State[best_candidate]>0)+1
                Q[position] = Summ[position] / (numpy.sqrt((A[:position+1] ** 2).sum()) * numpy.sqrt(position+1))

                #check
                #keypoints, zebra = self.get_path_back(position, Summ, Direction, State)
                #Summ1 = A[zebra == +1].sum() - A[zebra == -1].sum()
                #Summ2 = numpy.dot(A, zebra)
                #Q2 = numpy.corrcoef(zebra[:position+1], A0[:position+1])[0, 1]


        keypoints, zebra = [], None

        idx = numpy.where(abs(State) == n_periods)
        while n_periods>=2 and len(idx[0])==0:
            n_periods-=1
            idx = numpy.where(abs(State) == n_periods)

        if len(idx[0])>0:
            end = idx[0][numpy.argmax(Q[idx])]
            keypoints,zebra = self.get_path_back(end,Summ,Direction,State)
            keypoints = numpy.flip(keypoints)
            Q_best = numpy.abs(Q[end])
            Q_check = numpy.corrcoef(zebra[:end], A0[:end])[0, 1]

        # if do_debug:
        #     if zebra is not None:
        #         zebra256 = zebra.copy()
        #         zebra256[zebra > 0] = 255 - 64
        #         zebra256[zebra < 0] = 64
        #         zebra256[zebra == 0] = -10
        #     else:
        #         zebra256 = None
        #
        #     image_signal = tools_draw_numpy.draw_signals([A0, zebra256, 128*(Q+1)], keypoints)
        #     if debug_flip:image_signal = cv2.flip(image_signal, 1)
        #     cv2.imwrite(self.folder_out + base_name + '_path.png', image_signal)

        return keypoints
# ----------------------------------------------------------------------------------------------------------------
    def gen_zebra(self, duration, zero_point, n_periods):

        t = numpy.linspace(0, 0.5, duration)
        sq = signal.square(2 * numpy.pi * n_periods * t)

        res = numpy.zeros(duration)
        if zero_point is not None:
            res[zero_point:] = sq[:duration - zero_point]
            res[numpy.arange(zero_point, 0, -1)] = -sq[:zero_point]
        else:
            res = sq

        res[res > 0] = 192
        res[res <= 0] = 64
        return res
# ----------------------------------------------------------------------------------------------------------------
    def get_period_step_supervised(self, signal, the_range, x_phase0, base_name=None, do_debug=False):

        # X, success = tools_IO.load_if_exists(self.folder_out + 'cache/' + base_name + '_period_step.dat')
        # if (not do_debug) and success:
        #     return X

        Q1 = numpy.full(len(signal), -1, dtype=numpy.float32)
        Q2 = numpy.full(len(signal), -1, dtype=numpy.float32)
        values = []
        for period_step in the_range:

            start1, stop1,value1 = x_phase0-period_step,x_phase0,0
            if start1>=0 and signal[start1]>=0:
                S1 = signal[start1:stop1]
                Z1 = numpy.full(len(S1),+128+64,dtype=numpy.float32)
                Z1[:len(Z1)//2]=128-64
                value1 = (numpy.abs(numpy.corrcoef(Z1, S1)[0, 1])) * 255
                Q1[x_phase0 - period_step] = value1

            start2, stop2, value2 = x_phase0, x_phase0+period_step,0
            if stop2<len(signal) and signal[stop2]>=0:
                S2 = signal[start2:stop2]
                Z2 = numpy.full(len(S2), +128+64, dtype=numpy.float32)
                Z2[:len(Z2) // 2] = 128-64
                value2 = (numpy.abs(numpy.corrcoef(Z2,S2)[0, 1])) * 255
                Q2[x_phase0 + period_step] = value2

            values.append(max(value1,value2))

        period_step = int(0.5*the_range[int(numpy.argmax(values))])
        # tools_IO.write_cache(self.folder_out + 'cache/' + base_name + '_period_step.dat', period_step)
        #
        # if do_debug:
        #     cv2.imwrite(self.folder_out + '_period_search.png',tools_draw_numpy.draw_signals([signal,Q1,Q2],[x_phase0,x_phase0-period_step,x_phase0+period_step]))

        return period_step
# ----------------------------------------------------------------------------------------------------------------
    def get_period_step_auto(self, signal, the_range, base_name=None, do_debug=False):

        start = 0
        if numpy.any(signal < 0):start = numpy.argmax(signal >= 0)
        if numpy.any(signal[start:] < 0):stop = start + numpy.argmax(signal[start:] < 0)
        else:stop = len(signal)

        cand, Q, mids = [],[],[]
        #for mid in [(start+stop)//2]:
        for mid in range((start+stop)//2-the_range[0]//2,(start+stop)//2+the_range[0]//2,the_range[0]//4):
            q = []
            for period_step in the_range:
                start1,stop1 = mid,mid+period_step
                start2,stop2 = mid-period_step,mid
                if stop1 >=len(signal) or signal[stop1]<0 or start2 < 0 or signal[start2] < 0:value = 0
                else:value = (numpy.corrcoef(signal[start1:stop1], signal[start2:stop2])[0, 1]) * 255
                q.append(value)

            best = int(numpy.argmax(q))
            Q.append(q[best])
            cand.append(int(0.5 * the_range[best]))
            mids.append(mid)

        #period_step = numpy.dot(cand,Q)/numpy.sum(Q)
        best = int(numpy.argmax(Q))
        mid  = mids[best]
        period_step = cand[best]


        # if do_debug:
        #     cv2.imwrite(self.folder_out + base_name+ '_period_search_auto.png',tools_draw_numpy.draw_signals([signal],[mid,mid-period_step,mid+period_step]))

        return period_step
# ----------------------------------------------------------------------------------------------------------------
    def get_positions(self, signal, step_min, step_max, x_fixed, n_periods, search_leftwards=True, base_name=None, do_debug=False):

        start = max(0,x_fixed-n_periods*step_max)
        if numpy.any(signal[start:] < 0):
            start+= numpy.argmax(signal[start:] >= 0)
        if numpy.any(signal[start:] < 0):
            stop = start + numpy.argmax(signal[start:] < 0)
        else:
            stop = min(len(signal), x_fixed + n_periods * step_max)

        if search_leftwards:
            S_left  = signal[start:x_fixed]
            path = self.get_best_path_zebra(numpy.flip(S_left), step_min, step_max, n_periods,base_name,debug_flip=True, do_debug=do_debug)
            path = [start+len(S_left)-1-each for each in path][1:-1]
        else:
            S_right = signal[x_fixed:stop]
            path = self.get_best_path_zebra(S_right, step_min, step_max, n_periods, base_name, debug_flip=False,do_debug=do_debug)
            path = [x_fixed + each for each in path][1:-1]

        return path
# ----------------------------------------------------------------------------------------------------------------
    def prepare_signal(self, gray2d, image_mask, th_up, do_histo=True):

        result = []
        for c in range(gray2d.shape[1]):
            idx = numpy.where(image_mask[th_up:,c] > 0)
            if len(idx[0]>0):
                array = gray2d[th_up+idx[0],c-2:c+2]
                if do_histo:
                    hist, bins = numpy.histogram(array, 256, [0, 256])
                    idx = numpy.argsort(-hist)
                    values = bins[idx[:7]]
                    value = values.mean()
                else:
                    value = array.mean()
            else:
                value = -10
            result.append(value)

        result = numpy.array(result)

        result2d = 0*gray2d
        for c in range(gray2d.shape[1]):
            if result[c]>=0:
                result2d[:,c] = result[c]

        return result, result2d
# ----------------------------------------------------------------------------------------------------------------
    def prepare_signals(self,image,h_ipersp,do_debug = False):

        self.T.Tic('warpPerspective')
        image[image==0]=1
        image_wrapped = cv2.warpPerspective(image, h_ipersp, (self.GT_data.GT_width, self.GT_data.GT_height), borderValue=(0, 0, 0))
        image_mask = tools_image.desaturate_2d(image_wrapped)
        image_mask[image_mask>0]=1
        gray2d = tools_image.desaturate_2d(image_wrapped)

        self.T.Tic('prepare_signal_1')
        signal_short, signal2d = self.prepare_signal(gray2d, image_mask, th_up = int(gray2d.shape[0] * 0.65))
        self.T.Tic('prepare_signal_2')
        signal, signal2d = self.prepare_signal(gray2d, image_mask, 0)

        # if do_debug:
        #     cv2.imwrite(self.folder_out + '_wrapped_gray.png', image_wrapped)
        #     cv2.imwrite(self.folder_out + '_wrapped_sig.png', signal2d)

        return signal, signal_short
# ----------------------------------------------------------------------------------------------------------------
    def extract_vanishing_lines(self,image, L, base_name=None,do_debug=False):
        base_name = "base_"
        # X, success = tools_IO.load_if_exists(self.folder_out + 'cache/' + base_name + '_lines_van.dat')
        # if (not do_debug) and success:
        #     L.lines_van = X
        #     return L.lines_van


        n_periods = 6
        if L.point_van is None or L.line_upper_bound is None:return []
        if (L.line_midfield is None) and (L.line_side_L is None) and (L.line_side_R is None): return []

        self.T.Tic('get_inverce_perspective_mat')
        h_ipersp = tools_render_CV.get_inverce_perspective_mat(image, L.point_van, L.line_upper_bound,L.line_lower_bound, self.GT_data.GT_width, self.GT_data.GT_height)

        self.T.Tic('prepare_signals')
        signal, signal_short = self.prepare_signals(image,h_ipersp,do_debug=do_debug)

        x0, period_step = None,None

        if L.line_midfield is not None:
            self.T.Tic('line_midfield')
            line_midfield_trans = tools_pr_geom.perspective_transform(L.line_midfield.reshape((-1, 1, 2)).astype(numpy.float32), h_ipersp).reshape(4)
            x0 = int((line_midfield_trans[0] + line_midfield_trans[2]) / 2)
            #period_step = self.get_period_step_supervised(signal_short,range(50, 300),x0,base_name)
            period_step = self.get_period_step_auto(signal_short, range(50, 300), base_name, do_debug=do_debug)

            positions_L = self.get_positions(signal, period_step - 10, period_step + 15, x0, n_periods,search_leftwards=True, base_name=base_name+'_L_',do_debug=do_debug)
            positions_R = self.get_positions(signal, period_step - 10, period_step + 15, x0, n_periods,search_leftwards=False,base_name=base_name+'_R_',do_debug=do_debug)
            idx = self.GT_data.strips_per_field//2-1
            if len(positions_L)>0:
                lines = numpy.array([(pos, 0, pos, self.GT_data.GT_height) for pos in reversed(positions_L)], dtype=numpy.float32)
                L.lines_van[idx-len(lines):idx]=(cv2.perspectiveTransform(lines.reshape((-1, 1, 2)), numpy.linalg.inv(h_ipersp)).reshape((-1, 4)))

            if len(positions_R) > 0:
                lines = numpy.array([(pos, 0, pos, self.GT_data.GT_height) for pos in positions_R], dtype=numpy.float32)
                L.lines_van[idx:idx+len(lines)] = (cv2.perspectiveTransform(lines.reshape((-1, 1, 2)), numpy.linalg.inv(h_ipersp)).reshape((-1, 4)))

        elif L.line_side_L is not None:
            self.T.Tic('line_side_L')
            line_side_trans = tools_pr_geom.perspective_transform(L.line_side_L.reshape((-1, 1, 2)).astype(numpy.float32), h_ipersp).reshape(4)
            x0 = int((line_side_trans[0] + line_side_trans[2]) / 2)
            period_step = self.get_period_step_auto(signal_short, range(50, 300), base_name, do_debug=do_debug)
            positions = [x0] + self.get_positions(signal, period_step - 20, period_step + 15, x0, n_periods,search_leftwards=False, base_name=base_name,do_debug=do_debug)
            idx = (self.GT_data.strips_per_field//2)//3-1   #=2
            if len(positions)>0:
                lines = numpy.array([(pos, 0, pos, self.GT_data.GT_height) for pos in positions], dtype=numpy.float32)
                L.lines_van[idx:idx+len(lines)]=(cv2.perspectiveTransform(lines.reshape((-1, 1, 2)), numpy.linalg.inv(h_ipersp)).reshape((-1, 4)))

        elif L.line_side_R is not None:
            self.T.Tic('line_side_R')
            line_side_trans = tools_pr_geom.perspective_transform(L.line_side_R.reshape((-1, 1, 2)).astype(numpy.float32), h_ipersp).reshape(4)
            x0 = int((line_side_trans[0] + line_side_trans[2]) / 2)
            period_step = self.get_period_step_auto(signal_short, range(50, 300), base_name, do_debug=do_debug)
            positions = self.get_positions(signal, period_step - 20, period_step + 15, x0, n_periods,search_leftwards=True, base_name=base_name) + [x0]
            idx = 13
            if len(positions) > 0:
                lines = numpy.array([(pos, 0, pos, self.GT_data.GT_height) for pos in positions],dtype=numpy.float32)
                L.lines_van[idx - len(lines):idx] = (cv2.perspectiveTransform(lines.reshape((-1, 1, 2)), numpy.linalg.inv(h_ipersp)).reshape((-1, 4)))

        L.cut_vanishing()
        self.T.Tic('IO')
        # if do_debug:
        #     cv2.imwrite(self.folder_out + base_name + '_signal.png', tools_draw_numpy.draw_signals([signal], [x0,x0-period_step,x0+period_step]))
        #
        # tools_IO.write_cache(self.folder_out + 'cache/' + base_name + '_lines_van.dat', L.lines_van)
        # self.T.stage_stats(self.folder_out + 'log_extract_vanishing_lines.txt')
        return L.lines_van
# ----------------------------------------------------------------------------------------------------------------
