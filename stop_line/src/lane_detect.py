import cv2
import numpy as np

class LaneDetector:

    def __init__(self):
        self.kernel_size = 5

        self.low_threshold = 0
        self.high_threshold = 50

    def detect(self, img):
        h,w = img.shape

        # Blur image
        blur = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)

        # Canny transform to get edges
        edges = cv2.Canny(blur, self.low_threshold, self.high_threshold)
        debug = cv2.cvtColor( edges, cv2.COLOR_GRAY2BGR );

        lines = self.detectCloseLines(edges, debug)

        edges_along_line = self.keepIfAlongLines(edges, lines, extrapolate=True)

        lane_segments = self.detectLaneSegments(edges_along_line, debug)
        lane_edges = self.keepIfAlongLines(edges, lane_segments, extrapolate=False)

        lanes = self.detectLanes(lane_edges, debug)
        stop_line = self.lanesToStopLine(lanes, debug)
        # cv2.imshow('debug1', debug)
        # cv2.waitKey(0)

        return stop_line, debug

    def lanesToStopLine(self, lanes, debug):
        left_points = []
        left_lanes = []
        right_points = []
        right_lanes = []
        for lane in lanes:
            p1,p2 = lane
            theta = self.computeTheta(p1,p2)
            if theta > 0:
                right_points += [p1,p2]
                right_lanes.append(lane)
            else:
                left_points += [p1,p2]
                left_lanes.append(lane)

        if len(left_points) == 0 or len(right_points) == 0:
            # No stop line
            return None

        left_points.sort(key=lambda x: x[1])
        right_points.sort(key=lambda x: x[1])
        left_min = left_points[0]
        right_min = right_points[0]

        x_avg = (left_min[0]+right_min[0])/2
        y_avg = (left_min[1]+right_min[1])/2
        p_avg = (x_avg, y_avg)

        p1 = self.evaluate_segment_at_y(y_avg, left_lanes[0])
        p2 = self.evaluate_segment_at_y(y_avg, right_lanes[0])
        stop_line = (p1,p2)

        # cv2.line(debug, p1, p2, (128, 255, 128), 2)
        return stop_line



    def detectLanes(self, edges, debug):
        rho = 1
        theta = np.pi/180
        threshold = 50
        min_line_length = 100
        max_line_gap = 100
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
        
        line_segments = []
        if lines is not None:
            a,b,c = lines.shape
            for i in range(a):
                p1 = (lines[i][0][0], lines[i][0][1])
                p2 = (lines[i][0][2], lines[i][0][3])
                line_segments.append((p1,p2))
                cv2.line(debug, p1, p2, (255, 255, 0), 2)
        return line_segments


    def detectLaneSegments(self, edges, debug):
        h,w = edges.shape
        # Fit big lines
        rho = 1
        theta = np.pi/360
        threshold = 30
        min_line_length = 200
        max_line_gap = 100
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
        
        # Calculate lane slopes
        left_thetas = []
        right_thetas = []
        if lines is not None:
            a,b,c = lines.shape
            for i in range(a):
                p1 = (lines[i][0][0], lines[i][0][1])
                p2 = (lines[i][0][2], lines[i][0][3])
                theta = self.computeTheta(p1,p2)
                if theta > 0:
                    left_thetas.append(theta)
                else:
                    right_thetas.append(theta)

        left_theta = None
        right_theta = None
        if len(left_thetas) != 0:
            left_theta = sum(left_thetas)/len(left_thetas)
        if len(right_thetas) != 0:
            right_theta = sum(right_thetas)/len(right_thetas)

        # Find all small lines and filter
        lines = self.detectAllLines(edges)
        segments = []
        if lines is not None:
            a,b,c = lines.shape
            for i in range(a):
                p1 = (lines[i][0][0], lines[i][0][1])
                p2 = (lines[i][0][2], lines[i][0][3])
                theta = self.computeTheta(p1,p2)
                cv2.line(debug, p1, p2, (0, 255, 255), 2)
                if left_theta is not None and abs(theta - left_theta) < 0.2:
                    if p1[0] > w/2-25: #25 not good
                        cv2.line(debug, p1, p2, (0, 0, 255), 2)
                        segments.append((p1,p2))
                elif right_theta is not None and abs(theta - right_theta) < 0.2:
                    if p1[0] < w/2-25:
                        cv2.line(debug, p1, p2, (0, 0, 255), 2)
                        segments.append((p1,p2))
        return segments

    def computeTheta(self, p1, p2):
        slope = (p1[1]-p2[1])*1.0/(p1[0]-p2[0]+0.01)# Fix divide by zero
        return np.arctan2(slope, 1)

    def detectAllLines(self, edges):
        rho = 1
        theta = np.pi/360
        threshold = 2
        min_line_length = 5
        max_line_gap = 0
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
        return lines

    def keepIfAlongLines(self, edges, lines, extrapolate=False):
        mask = np.zeros(edges.shape)
        h,w = edges.shape
        for line in lines:
            p1,p2 = line
            if extrapolate:
                p1 = np.array((p1[0], p1[1]))
                p2 = np.array((p2[0], p2[1]))
                v = p1 - p2
                p1 = v*100 + p1
                p2 = v*-100 + p2
                p1 = (p1[0], p1[1])
                p2 = (p2[0], p2[1])
            cv2.line(mask,p1,p2, 1, 30)

        horizon = int(0.6*h)
        mask[:horizon,:] = 0
        masked = np.multiply(edges,mask)
        return masked.astype(np.uint8)

    def detectCloseLines(self, edges, debug):
        h,w = edges.shape
        # Mask to focus on bottom of image
        mask_percent = 0.85
        mask = np.zeros((h,w))
        left = int(.1*w)
        right = int(.9*w)
        horizon = int(mask_percent*h)
        mask[horizon:, left:right] = 1
        mask = mask.astype(np.uint8)

        masked_edges = np.multiply(edges, mask)

        # Draw horizon
        # cv2.line(debug, (0,horizon), (w, horizon), (0,255,0), 2)

        rho = 1
        theta = np.pi/180
        threshold = 10
        min_line_length = 30
        max_line_gap = 100
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
        
        line_segments = []
        if lines is not None:
            a,b,c = lines.shape
            for i in range(a):
                p1 = (lines[i][0][0], lines[i][0][1])
                p2 = (lines[i][0][2], lines[i][0][3])
                line_segments.append((p1,p2))
                cv2.line(debug, p1, p2, (255, 0, 0), 2)

        return line_segments

    def evaluate_segment_at_y(self, y, segment):
        p1,p2 = segment
        slope = (p1[1]-p2[1])*1.0/(p1[0]-p2[0]+0.01) + 0.01

        dy = y - p1[1]
        dx = dy / slope
        print slope
        print dx
        print dy
        return int(p1[0] + dx), int(p1[1] + dy)



if __name__=="__main__":
    img = cv2.imread("images/street4.png", 0)

    detector = LaneDetector()

    detector.detect(img)

