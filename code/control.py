import math
from collections import defaultdict

import cv2
import numpy as np
import pyAdb
import pytesseract
import random

circle_debug = False
text_debug = False
log = True


class Atom:
    atom_num = None
    atom_name = ''
    atom_pos = (0, 0)

    atom_dict = defaultdict(lambda: 'Unknown atom',
                            {-3: 'black_p', -2: 'white_c', -1: 'blue_m', 0: 'red_p', 1: 'H', 2: 'He', 3: 'Li', 4: 'Be',
                             5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si',
                             15: 'P', 16: 'S',
                             17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn',
                             26: 'Fe', 27: 'Co',
                             28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se'})

    def __init__(self, no: int, pos):
        self.atom_num = no
        self.atom_name = self.atom_dict[no]
        self.atom_pos = pos

    def __str__(self):
        return f'{self.atom_num}:{self.atom_name}'


class AGame:
    """Game class, each instance represents a game"""
    score = 0
    center_atom = None
    atoms_list = []
    state = 'gaming'

    # Game screenshot related parameters
    screen_img = None

    def __init__(self):
        """"""
        self.update()

    def __str__(self):
        return f'The game state is:{self.state} \n' \
               f'score={self.score} \n' \
               f'center atom is: {self.center_atom} \n' \
               f'around atoms are:{[str(atom) for atom in self.atoms_list]}'

    def update(self):
        """get the information """
        # get the screen image
        pyAdb.screen_cap('tmp')
        self.screen_img = cv2.imread('../img_tmp/tmp.png')  # shape:(1600, 900 ,3)
        self.screen_img = cv2.resize(self.screen_img, (900 // 2, 1600 // 2))

        # get the score
        self.score = self.get_score(self.screen_img)

        # get the atom_list
        self.atoms_list = self.get_atoms_list(self.screen_img)
        self.center_atom = self.atoms_list.pop(0)

    def random_play(self):
        index = random.randint(0, len(self.atoms_list)-1)
        if self.center_atom.atom_name == 'blue_m':
            self.catch_index(index)
        elif self.center_atom.atom_name == 'white_c':
            self.catch_index(index)
        else:
            self.shoot_after_index(index)
        self.update()

    def shoot_after_index(self, index):
        """shoot the center atom after the index atom"""
        atom_num = len(self.atoms_list)
        x1, y1 = self.atoms_list[index].atom_pos
        if atom_num == 1:
            x, y = -x1, -y1
        elif atom_num == 2:
            x, y = x1, y1
        else:
            x2, y2 = self.atoms_list[(index + 1) % atom_num].atom_pos
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
        x = (x + 224) * 2
        y = (y + 219) * 2 + 450
        pyAdb.click(x, y)

    def catch_index(self, index):
        """catch the index atom"""
        x, y = self.atoms_list[index].atom_pos
        x = (x + 224) * 2
        y = (y + 219) * 2 + 450
        pyAdb.click(x, y)

    @classmethod
    def get_score(cls, screen_img):
        """get the score from the screen_img"""
        score_img = screen_img[75:150, :]
        gray_score_img = cv2.cvtColor(score_img, cv2.COLOR_BGR2GRAY)
        _, score_img = cv2.threshold(gray_score_img, 127, 255, cv2.THRESH_BINARY)
        score = pytesseract.image_to_string(gray_score_img, lang="num", config='--psm 7')
        return int(score)

    @classmethod
    def get_atoms_list(cls, screen_img):
        """get the atom from the screen_img"""
        atoms_list = []
        tmp_image = cv2.imread('../img_tmp/tmp.png')  # shape:(1600, 900 ,3)

        # image preprocess
        # clean the back ground
        for i in range(225, 675):
            for j in range(450):
                b, g, r = screen_img[i, j, :]
                if 35 < b < 65 and 35 < g < 65 and 75 < r < 100:
                    screen_img[i, j, :] = 255, 0, 0

        # cut the screen
        atoms_img = screen_img[225:675, :]
        tmp_image = tmp_image[450:1350, :]

        # to gray
        gray_atoms_img = cv2.cvtColor(atoms_img, cv2.COLOR_BGR2GRAY)
        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2GRAY)

        # find the circle by Hough change
        circles = cv2.HoughCircles(gray_atoms_img, cv2.HOUGH_GRADIENT, 1, 30, np.array([]), 1, 15, 27, 28)

        # show the result
        if circle_debug:
            if circles is not None:
                tmp_img = np.copy(gray_atoms_img)
                _, b, _ = circles.shape
                for i in range(b):
                    cv2.circle(tmp_img, (circles[0][i][0], circles[0][i][1]), int(circles[0][i][2]), (0, 0, 255), 2)
                    cv2.circle(tmp_img, (circles[0][i][0], circles[0][i][1]), 2, (255, 255, 0),
                               3)  # draw center of circle
                cv2.imshow("detected circles", tmp_img)
            cv2.waitKey(0)

        # sort the atoms by the angle and distance
        circles = circles[0].tolist()
        center = []
        if circles is not None:
            circles.sort(key=lambda cir: (cir[0] - 225) ** 2 + (cir[1] - 225) ** 2)
            center.append(circles.pop(0))
            circles.sort(key=AGame.angle)
            circles = center + circles

            # get the atom_num and pos
            for circle in circles:
                x, y = int(circle[0]), int(circle[1])
                # atom = gray_atoms_img[y - 15:y + 22, x - 13:x + 13]
                atom = tmp_image[y * 2 - 30:y * 2 + 44, 2 * x - 26:2 * x + 26]

                no_img = atom[50:, :]

                no = pytesseract.image_to_string(no_img, lang='num', config='--psm 7')

                tmp_mean = np.mean(atoms_img[y - 15:y + 22, x - 12:x + 12], 0)
                bgr_atom = np.mean(tmp_mean, 0)

                # judge exception atom like red-plus black-plus
                b, g, r = bgr_atom
                exc_dict = {'red_p': {'b': 67, 'g': 73, 'r': 207, 'no': 0},
                            'blue_m': {'b': 195, 'g': 121, 'r': 67, 'no': -1},
                            'white_c': {'b': 224, 'g': 223, 'r': 230, 'no': -2},
                            'black_m': {'b': 10, 'g': 9, 'r': 12, 'no': -3}}
                for key in exc_dict.keys():
                    if exc_dict[key]['b'] * 0.9 < b < exc_dict[key]['b'] * 1.1 \
                            and exc_dict[key]['g'] * 0.9 < g < exc_dict[key]['g'] * 1.1 \
                            and exc_dict[key]['r'] * 0.9 < r < exc_dict[key]['r'] * 1.1:
                        name = key
                        no = exc_dict[name]['no']
                atoms_list.append(Atom(int(no), (x - 224, y - 219)))
            if text_debug:
                for atom in atoms_list:
                    print(atom)
        return atoms_list

    @staticmethod
    def angle(circle):
        """Calculate the angle between atom and central atom"""
        x, y, _ = circle
        x, y = x - 224, y - 219
        result = 0
        if x > 0 and y > 0:
            result = math.atan(y / x)
        if x < 0 < y:
            result = math.pi * 0.5 + math.atan(-x / y)
        if x < 0 and y < 0:
            result = math.pi + math.atan(y / x)
        if x > 0 > y:
            result = math.pi * 1.5 + math.atan(-x / y)
        return result


if __name__ == "__main__":
    a_game = AGame()
    print(a_game)
    while True:
        a_game.random_play()
