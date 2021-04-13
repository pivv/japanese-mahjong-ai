import tensorflow as tf

import hashlib

from .mahjong_log_pb2 import MahjongLog

class MahjongLogIO(object):
    def __init__(self, file_path = None):
        self._file_path = file_path
        self._read_count = 0
        self._write_count = 0

    def set_path(self, file_path):
        self._file_path = file_path
        self._read_count = 0
        self._write_count = 0

    def serialized_read(self, file_path = None):
        if not self._file_path:
            assert (file_path is not None)
            reader = tf.python_io.tf_record_iterator(file_path)
        else: # recursively reading
            if self._read_count == 0:
                self._reader = tf.python_io.tf_record_iterator(self._file_path)
            reader = self._reader
            self._read_count += 1
        for serialized_log in reader:
            yield serialized_log

    def read(self, file_path = None):
        if not self._file_path:
            assert (file_path is not None)
            reader = tf.python_io.tf_record_iterator(file_path)
        else: # recursively reading
            if self._read_count == 0:
                self._reader = tf.python_io.tf_record_iterator(self._file_path)
            reader = self._reader
            self._read_count += 1
        for serialized_log in reader:
            yield MahjongLog.FromString(serialized_log)

    def write(self, data, file_path = None):
        if not self._file_path:
            assert (file_path is not None)
            writer = tf.python_io.TFRecordWriter(file_path)
        else: # recursively writing
            if self._write_count == 0:
                self._writer = tf.python_io.TFRecordWriter(self._file_path)
            writer = self._writer
            self._write_count += 1
        writer.write(data.SerializeToString())

