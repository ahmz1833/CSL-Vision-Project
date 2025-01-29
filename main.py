import os
import re
import cv2
import time
import math
import numpy as np
import tkinter as tk

import requests


class ImageProcessor:
    """
        The ImageProcessor class is designed to detect and measure the area of various shapes in a live stream image.
        The class captures images from a camera feed or processes images from a file or URL, applies image processing techniques to identify contours of shapes, and calculates their areas.
        The results are then displayed with contours and relevant information on the screen.
        Attributes:
            image_path (str): Path to the input image. If empty, the camera will be used. it can be also url of image
            surrounding_color (tuple): RGB color for contour highlights.
            rectangle_color (tuple): RGB color for rectangles highlights.
            show_rectangle (bool): Whether to display bounding rectangles around contours.
            threshold1 (int): First threshold for edge detection.
            threshold2 (int): Second threshold for edge detection.
            min_area (int): Minimum area of contours to be processed.
            standard_area (int): Area normalization factor for labeling contours.
            frame_width (int): Width of frame used for processing.
            frame_height (int): Height of frame used for processing.
            max_line_gap (int): Max line gap between lines.
            min_line_length (int): Minimum line length for line detection.
            threshold3 (int): Threshold for line detection.
            line_color (tuple): RGB color for line detection.
    """

    """
        Key mappings for user inputs
        GO_TO_DEBUG: Enter debug mode (key 'd')
        GO_TO_MAIN: Return to main mode (key 'm')
        GO_TO_SELECT: Enter select mode (key 's')
        GO_TO_EXIT: Exit the application (key 'q')
        GO_RIGHT: Move to the next contour (key 'r')
        GO_LEFT: Move to the previous contour (key 'l')
        ENTER: Confirm action and apply standard value (Enter key)
    """
    GO_TO_DEBUG = 'd'
    GO_TO_MAIN = 'm'
    GO_TO_SELECT = 's'
    GO_TO_LINE_DETECTION = 'n'
    GO_TO_EXIT = 'q'
    GO_RIGHT = 'r'
    GO_LEFT = 'l'
    ENTER = 0x0D

    def __init__(self, image_path='', surrounding_color=(50, 50, 50), rectangle_color=(255, 0, 255),
                 show_rectangle=True, frame_width=640, frame_height=480, capture_device=0, threshold1=23, threshold2=20,
                 min_area=20, standard_area=1, min_line_length=100, max_line_gap=10, threshold3=100,
                 line_color=(0, 255, 0)):
        """
            Initializes the ImageProcessor class with default or user-defined parameters.
            Args:
                image_path (str): Path to the input image or empty for camera.
                surrounding_color (tuple): RGB color for contour highlights.
                rectangle_color (tuple): RGB color for bounding rectangles.
                show_rectangle (bool): Whether to show rectangles.
                frame_width (int): Width of the video frame.
                frame_height (int): Height of the video frame.
                capture_device (int): Camera index for video capture.
                threshold1 (int): First threshold for Canny edge detection.
                threshold2 (int): Second threshold for Canny edge detection.
                min_area (int): Minimum contour area to process.
                standard_area (int): Area normalization factor for labeling contours.
                min_line_length (int): Minimum line length for line detection.
                max_line_gap (int): Maximum line gap for line detection.
                threshold3 (int): Threshold for line detection.
                line_color (tuple): RGB color for line detection.
        """

        # Initialize instance variables
        self.image_path = image_path
        self.surrounding_color = surrounding_color
        self.rectangle_color = rectangle_color
        self.show_rectangle = show_rectangle
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.min_area = min_area
        self.standard_area = standard_area
        self.first_show = True
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.threshold3 = threshold3
        self.line_color = line_color

        url_pattern = re.compile(
            r'^(https?|ftp)://'
            r'([A-Z0-9][A-Z0-9_-]*(?:\.[A-Z0-9][A-Z0-9_-]*)+):?'
            r'(:\d+)?/?'
            r'([-A-Z0-9+&@#/%=~_|$?!:,.]*)$',
            re.IGNORECASE
        )
        self.is_url = re.match(url_pattern, self.image_path) is not None

        # Determine if using camera or image file
        self.have_camera = (image_path == '')

        # Set up video capture if using camera
        self.cap = cv2.VideoCapture(capture_device)
        self.cap.set(3, frame_width)
        self.cap.set(4, frame_height)

    def __change_parameter(self, parameter, value):
        """
            Updates a parameter dynamically during runtime.
            Args:
                parameter (str): The name of the parameter to update.
                value (any): The new value for the parameter.
        """
        setattr(self, parameter, value)

    def __show_parameters_window(self):
        """
            Creates a parameter adjustment window with trackbars for real-time tuning.
        """

        def create_trackbar(name, max_value):
            cv2.createTrackbar(name, "Parameters", getattr(self, name.lower()), max_value,
                               lambda v: self.__change_parameter(name.lower(), v))

        cv2.namedWindow("Parameters")
        cv2.resizeWindow("Parameters", 640, 270)
        create_trackbar("Threshold1", 255)
        create_trackbar("Threshold2", 255)
        create_trackbar("Min_Area", 100000)
        create_trackbar("Standard_Area", 1000000)
        create_trackbar("min_line_length", 255)
        create_trackbar("max_line_gap", 255)
        create_trackbar("threshold3", 255)

    def __show_image_centered(self, image, window_name="Window", window_width=None, window_height=None):
        """
            Displays a centered OpenCV window on the screen.
            Args:
                image (np.ndarray): Image to display.
                window_name (str): Title of the window.
                window_width (int, optional): Desired width of the window.
                window_height (int, optional): Desired height of the window.
        """

        def get_screen_resolution():
            root = tk.Tk()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.quit()
            return width, height

        cv2.imshow(window_name, image)
        if self.first_show:
            screen_width, screen_height = get_screen_resolution()
            if window_width is None or window_height is None:
                window_height, window_width, _ = image.shape
            x = int((screen_width - window_width) / 2)
            y = int((screen_height - window_height) / 2)
            cv2.moveWindow(window_name, x, y)
            self.first_show = False

    @staticmethod
    def __close_parameters_window():
        """
           Closes the parameter adjustment window.
        """
        try:
            cv2.destroyWindow("Parameters")
        except cv2.error:
            pass

    @staticmethod
    def __stack_images(scale, img_array):
        """
            Stacks images for display in a grid format.
            Args:
                scale (float): Scaling factor for images.
                img_array (tuple or list or array): 2D list of images to stack.
            Returns:
                np.ndarray: Stacked image.
        """
        rows = len(img_array)
        cols = len(img_array[0])
        rows_available = isinstance(img_array[0], list)
        width = img_array[0][0].shape[1]
        height = img_array[0][0].shape[0]
        if rows_available:
            for x in range(0, rows):
                for y in range(0, cols):
                    if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                        img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                    else:
                        img_array[x][y] = cv2.resize(img_array[x][y],
                                                     (img_array[0][0].shape[1], img_array[0][0].shape[0]), None, scale,
                                                     scale)
                    if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y],
                                                                                       cv2.COLOR_GRAY2BGR)
            image_blank = np.zeros((height, width, 3), np.uint8)
            hor = [image_blank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(img_array[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if img_array[x].shape[:2] == img_array[0].shape[:2]:
                    img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
                else:
                    img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                              scale)
                if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(img_array)
            ver = hor
        return ver

    def __get_con_line(self, img, img_result, show_all=True, line_detector_canny=None):
        """
            Processes and draws contours on the output image if line_detector_canny is null. otherwise it will draw lines
            Args:
                img (np.ndarray): Binary image for contour detection.
                img_result (np.ndarray): Image to draw contours on.
                show_all (bool, optional): Show all contours.
                line_detector_canny (cv2.Can): Canny for line detection.
            returns:
                0 -> means user wants to exit
                None -> means user wants to go to main window
        """

        def draw_contour(cnt):
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                # Draw contour and rectangle if area is greater than minimum area
                cv2.drawContours(img_result, [cnt], -1, self.surrounding_color, 2)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                if self.show_rectangle:
                    cv2.rectangle(img_result, (x, y), (x + w, y + h), self.rectangle_color, 2)
                cv2.putText(img_result, str(round(area / (self.standard_area if self.standard_area != 0 else 1), 2)),
                            (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 0, 0), 2)
                return True
            return False

        def draw_line(l):
            x1, y1, x2, y2 = l[0]
            cv2.line(img_result, (x1, y1), (x2, y2), self.line_color, 2)
            return True

        def handle_key_press():
            correct_key = True
            while True:
                if correct_key:
                    self.__show_image_centered(img_result, 'SELECT MODE: ' if not show_lines else 'LINE DETECTION MODE')
                    img_result[:] = img_copy[:]
                key = cv2.waitKey(0) & 0xFF
                if key == ord(self.GO_TO_EXIT):
                    return False
                elif key == ord(self.GO_RIGHT):
                    return 1
                elif key == ord(self.GO_LEFT):
                    return -1
                elif key == self.ENTER:
                    if show_lines:
                        x1, y1, x2, y2 = lines[index][0]
                        self.standard_area = math.ceil((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    else:
                        self.standard_area = math.ceil(cv2.contourArea(contours[index]))
                    return None
                else:
                    correct_key = False

        img_canny = line_detector_canny
        show_lines = line_detector_canny is not None
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        lines = cv2.HoughLinesP(img_canny, 1, np.pi / 180, self.threshold3, minLineLength=self.min_line_length,
                                maxLineGap=self.max_line_gap)
        img_copy = img_result.copy() if not show_all else None
        index, direction = 0, 1
        iterator = contours if not show_lines else lines
        while iterator is not None and index < len(iterator):
            if not show_lines:
                contour = iterator[index]
                condition = draw_contour(contour)
            else:
                line = iterator[index]
                condition = draw_line(line)

            if condition and not show_all:
                move = handle_key_press()
                if move is False:
                    return 0
                elif move is not None:
                    index = ((index + move) + len(iterator)) % len(iterator)
                    direction = move
                else:
                    break
            elif show_all:
                index += 1
            else:
                index = ((index + direction) + len(iterator)) % len(iterator)

    def __read_image(self, threshold1, threshold2):
        """
            Read and process the image, applying filters.
            args:
                threshold1 (float): Threshold of the first contour.
                threshold2 (float): Threshold of the second contour.
            returns:
                np.ndarray: Image read and processed.
        """
        # Capture image from camera or read from file
        if self.have_camera:
            success, img = self.cap.read()
            if not success:
                raise Exception("Unable to capture image from camera. Please check the camera connection or settings.")
            img = cv2.flip(img, 1)
        elif self.is_url is False:
            if not os.path.exists(self.image_path):
                raise Exception(f"File does not exist: {self.image_path}")

            img = cv2.imread(self.image_path)
            if img is None:
                raise Exception(f"Unable to read image from the file: {self.image_path}")
            img = cv2.resize(img, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)
        else:
            img = None
            tries = 1000
            for _ in range(tries):
                try:
                    response = requests.get(self.image_path, stream=True)
                    response.raise_for_status()
                    # Ensure the full content is downloaded
                    image_data = b""
                    for chunk in response.iter_content(chunk_size=8192):
                        image_data += chunk
                    image_data = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("Failed to decode image")
                    img = cv2.resize(img, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)
                    break
                except (requests.exceptions.RequestException, ValueError):
                    print(f"Unable to read image from the URL: attempt {_ + 1} times...")
                    time.sleep(0.0005)
                    if _ == tries - 1:
                        raise
        img_contour = img.copy()
        img_blur = cv2.GaussianBlur(img, (7, 7), 1)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        img_canny = cv2.Canny(img_gray, threshold1, threshold2)
        kernel = np.ones((5, 5))
        img_dil = cv2.dilate(img_canny, kernel, iterations=1)
        return img, img_blur, img_gray, img_canny, img_dil, img_contour

    @staticmethod
    def __get_key():
        """
           Captures and processes key inputs during runtime.
           Returns:
               int: The keycode of the pressed key, or 0 if no key is pressed.
        """
        key = cv2.waitKey(1) & 0xFF
        key_map = {
            ord(ImageProcessor.GO_TO_EXIT): 1,
            ord(ImageProcessor.GO_TO_MAIN): 2,
            ord(ImageProcessor.GO_TO_DEBUG): 3,
            ord(ImageProcessor.GO_TO_SELECT): 4,
            ord(ImageProcessor.GO_TO_LINE_DETECTION): 5
        }
        return key_map.get(key, 0)

    def __jump_destination(self, key):
        """
            Determines and jump to the next destination or operation based on the current input.
        """
        self.first_show = True
        ImageProcessor.__exit()
        actions = {2: self.__main, 3: self.__debug, 4: self.__select, 5: self.__line_detection}
        action = actions.get(key)
        if action:
            action()

    def __debug(self):
        """
            Activates debug mode for interactive parameter tuning and contour visualization.
        """
        show_parameters = True
        while True:
            img, img_blur, img_gray, img_canny, img_dil, img_contour = self.__read_image(self.threshold1,
                                                                                         self.threshold2)
            img_copy = img.copy()
            self.__get_con_line(img_dil, img_contour)
            self.__get_con_line(img_dil, img_copy, line_detector_canny=img_canny)
            img_stack = ImageProcessor.__stack_images(0.8,
                                                      ([img, img_blur, img_canny], [img_dil, img_copy, img_contour]))
            self.__show_image_centered(img_stack, 'DEBUG MODE: ')
            if show_parameters:
                self.__show_parameters_window()
                show_parameters = False
            key = ImageProcessor.__get_key()
            if key != 0: break
        ImageProcessor.__close_parameters_window()
        self.__jump_destination(key)

    def __main(self):
        """
            Main loop for processing images until a changing key is pressed. Handles image capturing, contour detection, and display.
        """
        while True:
            img, img_blur, img_gray, img_canny, img_dil, img_contour = self.__read_image(self.threshold1,
                                                                                         self.threshold2)
            self.__get_con_line(img_dil, img_contour)
            self.__show_image_centered(img_contour, 'MAIN MODE: ')
            key = ImageProcessor.__get_key()
            if key != 0: break
        self.__jump_destination(key)

    def __line_detection(self):
        img, img_blur, img_gray, img_canny, img_dil, img_contour = self.__read_image(self.threshold1,
                                                                                     self.threshold2)
        result = self.__get_con_line(img_dil, img_contour, False, line_detector_canny=img_canny)
        if result == 0:
            self.__jump_destination(1)
        else:
            self.__jump_destination(2)

    def __select(self):
        """
            The select function allows the user to iterate through detected contours in the live stream image and choose a specific contour to set as the standard area for further processing.
        """
        img, img_blur, img_gray, img_canny, img_dil, img_contour = self.__read_image(self.threshold1,
                                                                                     self.threshold2)
        result = self.__get_con_line(img_dil, img_contour, False)
        if result == 0:
            self.__jump_destination(1)
        else:
            self.__jump_destination(2)

    @staticmethod
    def __exit():
        cv2.destroyAllWindows()

    def start(self):
        welcome_message = """
                    * CSL-Vision-Project *
            ***************************************
            *                                     *
            *       Welcome to Area Detector      *
            *                                     *
            ***************************************
            Authors
                * AmirMahdi Tahmasebi - 402106178
                * AmirHossein Mirzaei - 402106661
                * MohammadMahdi Rajabi - 402106015
                * AmirHossein MohammadZadeh - 402106434
        """
        print(welcome_message)
        try:
            self.__main()
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    processor = ImageProcessor(
        show_rectangle=True,
        threshold1=230,
        threshold2=23,
        min_area=5,
        max_line_gap=4,
        threshold3=11)
    processor.start()
