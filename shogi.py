# -*- coding: utf-8 -*-

from chainer.datasets import tuple_dataset

import numpy as np


# 局面を表すクラス
class Shogi():

    # 初期化
    def __init__(self):
        self.board = np.zeros((9, 9), dtype=np.int32)
        self.hand = np.zeros((2, 8), dtype=np.int32)
        self.turn = 0
        for i in range(9):
            self.board[6, i] = 1
        self.board[8, 0] = self.board[8, 8] = 2
        self.board[8, 1] = self.board[8, 7] = 3
        self.board[8, 2] = self.board[8, 6] = 4
        self.board[8, 3] = self.board[8, 5] = 7
        self.board[7, 1] = 5
        self.board[7, 7] = 6
        self.board[8, 4] = 8
        for i in range(9):
            self.board[2, i] = -1
        self.board[0, 0] = self.board[0, 8] = -2
        self.board[0, 1] = self.board[0, 7] = -3
        self.board[0, 2] = self.board[0, 6] = -4
        self.board[0, 3] = self.board[0, 5] = -7
        self.board[1, 7] = -5
        self.board[1, 1] = -6
        self.board[0, 4] = -8

    # 一手指す
    def move(self, command):
        to_y = ord(command[3]) - ord('a')
        to_x = ord('9') - ord(command[2])
        if command[0].isdigit():
            from_y = ord(command[1]) - ord('a')
            from_x = ord('9') - ord(command[0])
            if self.board[to_y, to_x] != 0:
                self.hand[self.turn, abs(self.board[to_y, to_x]) % 8] += 1
            self.board[to_y, to_x] = self.board[from_y, from_x]
            self.board[from_y, from_x] = 0
            if len(command) == 5:
                self.board[to_y, to_x] += 8 if self.turn == 0 else -8
        else:
            piece = ['', 'P', 'L', 'N', 'S', 'G', 'B', 'R'].index(command[0])
            self.hand[self.turn, piece] -= 1
            self.board[to_y, to_x] = piece if self.turn == 0 else -piece
        self.turn = 1 - self.turn

    # 入力チャンネル
    def get_channels(self):
        channels = np.zeros((42, 9, 9), dtype=np.float32)
        for ii in range(9):
            for jj in range(9):
                if self.turn == 0:
                    i = ii
                    j = jj
                else:
                    i = 8 - ii
                    j = 8 - jj
                if self.board[i, j] > 0:
                    channels[self.board[i, j]-1, ii, jj] = 1
                if self.board[i, j] < 0:
                    channels[13-self.board[i, j], ii, jj] = 1
        for i in range(1, 8):
            channels[27+i, :, :] = self.hand[self.turn, i]
            channels[34+i, :, :] = self.hand[1-self.turn, i]
        return channels

    # 出力チャンネル
    def get_move_class(self, move):
        to_y = ord(move[3]) - ord('a')
        to_x = ord('9') - ord(move[2])
        if move[0].isdigit():
            from_y = ord(move[1]) - ord('a')
            from_x = ord('9') - ord(move[0])
            if self.turn == 1:
                to_y = 8 - to_y
                to_x = 8 - to_x
                from_y = 8 - from_y
                from_x = 8 - from_x
            if to_y == from_y:
                if to_x < from_x:
                    channel = 0
                else:
                    channel = 1
            elif to_x == from_x:
                if to_y < from_y:
                    channel = 2
                else:
                    channel = 3
            elif to_y - from_y == to_x - from_x:
                if to_y < from_y:
                    channel = 4
                else:
                    channel = 5
            elif to_y - from_y == from_x - to_x:
                if to_y < from_y:
                    channel = 6
                else:
                    channel = 7
            elif to_x < from_x:
                channel = 8
            else:
                channel = 9
            if len(move) == 5:
                channel += 10
        else:
            piece = ['', 'P', 'L', 'N', 'S', 'G', 'B', 'R'].index(move[0])
            channel = piece + 19
        return np.int32(channel*9*9+to_y*9+to_x)


# データの取得
def get_data(datasize):
    positions = []
    bestmoves = []
    with open('records2016_10818.sfen') as sfens:
        for sfen in sfens.readlines()[:datasize]:
            shogi = Shogi()
            for move in sfen.split()[2:-1]:
                positions.append(shogi.get_channels())
                bestmoves.append(shogi.get_move_class(move))
                shogi.move(move)

    train = tuple_dataset.TupleDataset(
        positions[:20*datasize], bestmoves[:20*datasize])
    test = tuple_dataset.TupleDataset(
        positions[20*datasize:], bestmoves[20*datasize:])
    return train, test
