import collections
import datetime
import cv2
import os
import logging
import argparse
from urllib.parse import urlparse
import paho.mqtt.client as mqtt

# Inspired by the article here:
# https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/


class MotionTracker:
    def __init__(self, callback=None):
        self.last_motion_detection = None
        self.motion_detected = False
        self.motion_started = None
        self.motion_stopped = None

        self.motion_start_timeout = datetime.timedelta(seconds=1)
        self.motion_events_threshold = 10
        self.motion_timeout = datetime.timedelta(seconds=5)
        self.frames_before_motion = datetime.timedelta(seconds=5)

        self.frames = collections.deque()
        self.motion_events = collections.deque()
        self.write_frames = False
        self.video_out = None
        self.callback = callback

    def update_motion(self, motion: bool, frame) -> None:
        now = datetime.datetime.now()

        self.frames.append((now, frame))
        while len(self.frames) and now - self.frames[0][0] > (self.frames_before_motion + self.motion_start_timeout):
            self.frames.popleft()

        while len(self.motion_events) and now - self.motion_events[0] > self.motion_start_timeout:
            self.motion_events.popleft()

        if motion:
            self.motion_events.append(now)
            self.last_motion_detection = now

            if not self.motion_detected and len(self.motion_events) >= self.motion_events_threshold:
                self.motion_detected = True

                save_dir = os.path.join("motion", now.strftime("%Y/%m/%d"))
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, now.strftime("%Y_%m_%d-%H_%M_%S.mp4"))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                height, width = frame.shape[:2]
                self.video_out = cv2.VideoWriter(save_path, fourcc, 12, (width, height), True)
                for dt, prev_frame in self.frames:
                    self.video_out.write(prev_frame)

                self.signal_motion_detected_changed()
                return

        if self.motion_detected:
            self.video_out.write(frame)
            delta = now - self.last_motion_detection
            if delta > self.motion_timeout:
                self.motion_detected = False
                self.motion_stopped = now
                self.motion_started = None
                self.video_out.release()
                self.video_out = None
                self.signal_motion_detected_changed()

    def signal_motion_detected_changed(self):
        logging.info("Motion %s", self.motion_detected and 'started' or 'stopped')
        self.callback(self.motion_detected)


class MotionDetector:
    def __init__(self, tracker, video_url, mask_filename=None, show_video=True, show_processing=False,
                 frame_display=None):
        self.tracker = tracker
        self.video_url = video_url
        self.last_frame = None
        self.update_last_frame = None
        self.exit = False
        self.show_video = show_video
        self.show_processing = show_processing
        self.frame_display = frame_display

        if mask_filename:
            self.mask = cv2.imread(mask_filename)
        else:
            self.mask = None

    def process(self):
        while not self.exit:
            logging.info("Starting video capture")
            capture = cv2.VideoCapture(self.video_url)
            try:
                self.__process_stream(capture)
            except KeyboardInterrupt:
                logging.info("Exiting...")
                self.exit = True
            except:
                logging.error("Error while processing frames", exc_info=True)
            capture.release()
            logging.info("Video Capture released")

    def __process_stream(self, capture):
        while True:
            ret, frame = capture.read()
            now = datetime.datetime.now()
            if not ret:
                break

            masked = cv2.bitwise_and(frame, self.mask) if self.mask is not None else frame
            gray = cv2.GaussianBlur(cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY), (21, 21), 0)

            if self.last_frame is None:
                self.last_frame = gray
                self.update_last_frame = now
                continue

            frame_delta = cv2.absdiff(self.last_frame, gray)
            thresholded = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]
            dilated = cv2.dilate(thresholded, None, iterations=2)
            contours = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            found_motion = False
            to_display = frame.copy()
            for contour in contours:
                if cv2.contourArea(contour) < 6000:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(to_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                found_motion = True

            if self.frame_display is not None:
                self.frame_display(to_display)
            self.tracker.update_motion(found_motion, to_display)

            if (now - self.update_last_frame).total_seconds() > 1:
                self.last_frame = gray
                self.update_last_frame = now

            if self.show_video:
                cv2.imshow('VIDEO', to_display)
                if self.show_processing:
                    cv2.imshow("Masked", masked)
                    cv2.imshow("Blurred & Gray scaled", gray)
                    cv2.imshow("delta", frame_delta)
                    cv2.imshow("thresholded", thresholded)
                    cv2.imshow("dilated", dilated)
                key = cv2.waitKey(1) & 0xFF
                # if the `q` key is pressed, break from the lop
                if key == ord("q"):
                    logging.info("Exiting on request")
                    self.exit = True
                    break


class MQTTNotifier:
    def __init__(self, client, topic):
        self.client = client
        self.topic = topic

    def motion_updated(self, motion_detected):
        self.client.publish(self.topic, payload=motion_detected and "motion start" or "motion stop")


class MQTTFrameDisplay:
    def __init__(self, client, topic):
        self.client = client
        self.topic = topic

    def frame_display(self, frame):
        success, data = cv2.imencode(".jpg", frame)
        if success:
            self.client.publish(self.topic, payload=data.tobytes())


def mqtt_client(url):
    url_details = urlparse(url)
    if url_details.scheme not in ('mqtt', 'mqtts'):
        raise RuntimeError('Unsupported URL for MQTT "%s"' % url)

    kwargs = {}
    if url_details.scheme == 'mqtts':
        kwargs['ssl'] = True
    if url_details.port is not None:
        kwargs['port'] = url_details.port
    if url_details.username is not None:
        kwargs['username'] = url_details.username
    if url_details.password is not None:
        kwargs['password'] = url_details.password

    client = mqtt.Client()
    client.connect(url_details.hostname, **kwargs)
    client.loop_start()
    return client


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Looks for motion in the video stream specified and takes short "
                                                 "snapshot videos of this motion, optionally posting an MQTT message")
    parser.add_argument("stream", help="URL of the video stream to process for motion")
    parser.add_argument("--mask", help="Image file to use to mask out parts of the image that you don't want to monitor"
                                       " for motion.")
    parser.add_argument("--show-video", action="store_true", help="Show the video in a window.")
    parser.add_argument("--show-processing", action="store_true", help="Show the processing steps in separate windows,"
                                                                       " needs --show-video")
    parser.add_argument("--mqtt", help="URL of the MQTT server to connect to send motion notifications to. "
                        "URLs are in the form mqtt[s]://<hostname>[:port]",
                        default=None)
    parser.add_argument("--topic", help="MQTT Topic to send motion notifications to.", default="motiondetector/state")
    parser.add_argument("--displaytopic",
                        help="MQTT Topic used to send realtime JPEG encoded frames from the video stream being"
                             " monitored",
                        default="")

    logging.basicConfig(format="%(asctime)-15s : %(levelname)s : %(message)s", level=logging.INFO)
    args = parser.parse_args()
    logging.info("Streaming from %s", args.stream)
    if args.mask:
        logging.info("Using mask from %s", args.mask)

    motion_callback = None
    display_frame = None

    if args.mqtt:
        client = mqtt_client(args.mqtt)
        motion_callback = MQTTNotifier(client, args.topic).motion_updated

        if args.displaytopic:
            display_frame = MQTTFrameDisplay(client, args.displaytopic).frame_display

    tracker = MotionTracker(motion_callback)
    detector = MotionDetector(tracker, args.stream, mask_filename=args.mask, show_video=args.show_video,
                              show_processing=args.show_processing, frame_display=display_frame)
    detector.process()
    cv2.destroyAllWindows()
