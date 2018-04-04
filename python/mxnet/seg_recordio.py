# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Read and write for the RecordIO data format."""
from __future__ import absolute_import
from collections import namedtuple

import ctypes
import struct
import numbers
import numpy as np

from .base import _LIB
from .base import RecordIOHandle
from .base import check_call
from .base import c_str
try:
    import cv2
except ImportError:
    cv2 = None

class MXSegRecordIO(object):
    """Reads/writes `RecordIO` data format, supporting sequential read and write.

    Example usage:
    ----------
    >>> record = mx.seg_recordio.MXSegRecordIO('tmp.rec', 'w')
    <mxnet.recordio.MXSegRecordIO object at 0x10ef40ed0>
    >>> for i in range(5):
    ...    record.write('record_%d'%i)
    >>> record.close()
    >>> record = mx.seg_recordio.MXSegRecordIO('tmp.rec', 'r')
    >>> for i in range(5):
    ...    item = record.read()
    ...    print(item)
    record_0
    record_1
    record_2
    record_3
    record_4
    >>> record.close()

    Parameters
    ----------
    uri : string
        Path to the record file.
    flag : string
        'w' for write or 'r' for read.
    """
    def __init__(self, uri, flag):
        self.uri = c_str(uri)
        self.handle = RecordIOHandle()
        self.flag = flag
        self.is_open = False
        self.open()

    def open(self):
        """Opens the record file."""
        if self.flag == "w":
            check_call(_LIB.MXRecordIOWriterCreate(self.uri, ctypes.byref(self.handle)))
            self.writable = True
        elif self.flag == "r":
            check_call(_LIB.MXRecordIOReaderCreate(self.uri, ctypes.byref(self.handle)))
            self.writable = False
        else:
            raise ValueError("Invalid flag %s"%self.flag)
        self.is_open = True

    def __del__(self):
        self.close()

    def close(self):
        """Closes the record file."""
        if not self.is_open:
            return
        if self.writable:
            check_call(_LIB.MXRecordIOWriterFree(self.handle))
        else:
            check_call(_LIB.MXRecordIOReaderFree(self.handle))
        self.is_open = False

    def reset(self):
        """Resets the pointer to first item.

        If the record is opened with 'w', this function will truncate the file to empty.

        Example usage:
        ----------
        >>> record = mx.seg_recordio.MXSegRecordIO('tmp.rec', 'r')
        >>> for i in range(2):
        ...    item = record.read()
        ...    print(item)
        record_0
        record_1
        >>> record.reset()  # Pointer is reset.
        >>> print(record.read()) # Started reading from start again.
        record_0
        >>> record.close()
        """
        self.close()
        self.open()

    def write(self, buf):
        """Inserts a string buffer as a record.

        Example usage:
        ----------
        >>> record = mx.seg_recordio.MXSegRecordIO('tmp.rec', 'w')
        >>> for i in range(5):
        ...    record.write('record_%d'%i)
        >>> record.close()

        Parameters
        ----------
        buf : string (python2), bytes (python3)
            Buffer to write.
        """
        assert self.writable
        check_call(_LIB.MXRecordIOWriterWriteRecord(self.handle,
                                                    ctypes.c_char_p(buf),
                                                    ctypes.c_size_t(len(buf))))

    def read(self):
        """Returns record as a string.

        Example usage:
        ----------
        >>> record = mx.seg_recordio.MXSegRecordIO('tmp.rec', 'r')
        >>> for i in range(5):
        ...    item = record.read()
        ...    print(item)
        record_0
        record_1
        record_2
        record_3
        record_4
        >>> record.close()

        Returns
        ----------
        buf : string
            Buffer read.
        """
        assert not self.writable
        buf = ctypes.c_char_p()
        size = ctypes.c_size_t()
        check_call(_LIB.MXRecordIOReaderReadRecord(self.handle,
                                                   ctypes.byref(buf),
                                                   ctypes.byref(size)))
        if buf:
            buf = ctypes.cast(buf, ctypes.POINTER(ctypes.c_char*size.value))
            return buf.contents.raw
        else:
            return None

class MXIndexedSegRecordIO(MXSegRecordIO):
    """Reads/writes `RecordIO` data format, supporting random access.

    Example usage:
    ----------
    >>> for i in range(5):
    ...     record.write_idx(i, 'record_%d'%i)
    >>> record.close()
    >>> record = mx.seg_recordio.MXIndexedSegRecordIO('tmp.idx', 'tmp.rec', 'r')
    >>> record.read_idx(3)
    record_3

    Parameters
    ----------
    idx_path : str
        Path to the index file.
    uri : str
        Path to the record file. Only supports seekable file types.
    flag : str
        'w' for write or 'r' for read.
    key_type : type
        Data type for keys.
    """
    def __init__(self, idx_path, uri, flag, key_type=int):
        self.idx_path = idx_path
        self.idx = {}
        self.keys = []
        self.key_type = key_type
        self.fidx = None
        super(MXIndexedSegRecordIO, self).__init__(uri, flag)

    def open(self):
        super(MXIndexedSegRecordIO, self).open()
        self.idx = {}
        self.keys = []
        self.fidx = open(self.idx_path, self.flag)
        if not self.writable:
            for line in iter(self.fidx.readline, ''):
                line = line.strip().split('\t')
                key = self.key_type(line[0])
                self.idx[key] = int(line[1])
                self.keys.append(key)

    def close(self):
        """Closes the record file."""
        if not self.is_open:
            return
        super(MXIndexedSegRecordIO, self).close()
        self.fidx.close()

    def seek(self, idx):
        """Sets the current read pointer position.

        This function is internally called by `read_idx(idx)` to find the current
        reader pointer position. It doesn't return anything."""
        assert not self.writable
        pos = ctypes.c_size_t(self.idx[idx])
        check_call(_LIB.MXRecordIOReaderSeek(self.handle, pos))

    def tell(self):
        """Returns the current position of write head.

        Example usage:
        ----------
        >>> record = mx.seg_recordio.MXIndexedSegRecordIO('tmp.idx', 'tmp.rec', 'w')
        >>> print(record.tell())
        0
        >>> for i in range(5):
        ...     record.write_idx(i, 'record_%d'%i)
        ...     print(record.tell())
        16
        32
        48
        64
        80
        """
        assert self.writable
        pos = ctypes.c_size_t()
        check_call(_LIB.MXRecordIOWriterTell(self.handle, ctypes.byref(pos)))
        return pos.value

    def read_idx(self, idx):
        """Returns the record at given index.

        Example usage:
        ----------
        >>> record = mx.seg_recordio.MXIndexedSegRecordIO('tmp.idx', 'tmp.rec', 'w')
        >>> for i in range(5):
        ...     record.write_idx(i, 'record_%d'%i)
        >>> record.close()
        >>> record = mx.seg_recordio.MXIndexedSegRecordIO('tmp.idx', 'tmp.rec', 'r')
        >>> record.read_idx(3)
        record_3
        """
        self.seek(idx)
        return self.read()

    def write_idx(self, idx, buf):
        """Inserts input record at given index.

        Example usage:
        ----------
        >>> for i in range(5):
        ...     record.write_idx(i, 'record_%d'%i)
        >>> record.close()

        Parameters
        ----------
        idx : int
            Index of a file.
        buf :
            Record to write.
        """
        key = self.key_type(idx)
        pos = self.tell()
        self.write(buf)
        self.fidx.write('%s\t%d\n'%(str(key), pos))
        self.idx[key] = pos
        self.keys.append(key)


ISegRHeader = namedtuple('HEADER', ['flag', 'label', 'image_size', 'label_size', 'id', 'id2'])
"""An alias for HEADER. Used to store metadata (e.g. labels) accompanying a record.
See mxnet.recordio.pack and mxnet.recordio.pack_img for example uses.

Parameters
----------
    flag : int
        Available for convenience, can be set arbitrarily.
    label : float or an array of float
        Typically used to store label(s) for a record.
    image_size: int
        length of image string.
    label_size: int
        length of label string.
    id: int
        Usually a unique id representing record.
    id2: int
        Higher order bits of the unique id, should be set to 0 (in most cases).
"""
_ISEGR_FORMAT = 'IfIIQQ'
_IR_SIZE = struct.calcsize(_ISEGR_FORMAT)

def pack(header, image_data, label_data):
    """Pack a string into MXImageRecord.

    Parameters
    ----------
    header : ISegRHeader
        Header of the image record.
    image_data : str
        Raw image string to be packed.
    label_data : str
        Raw label string to be packed.

    Returns
    -------
    s : str
        The packed string.

    Examples
    --------
    >>> id = 2574
    >>> img = cv2.imread(fullpath, cv2.IMREAD_COLOR)
    >>> ret, buf = cv2.imencode(".jpg", img)
    >>> assert ret, 'failed to encode image'
    >>> image_data = buf.tostring()
    >>> image_len = len(image_data)

    >>> label_path = item[-1]
    >>> label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    >>> ret, buf = cv2.imencode(".png", label)
    >>> assert ret, 'failed to encode label'
    >>> label_data = buf.tostring()
    >>> label_len = len(label_data)

    >>> header = mx.seg_recordio.ISegRHeader(0, 0, image_len, label_len, id, 0)
    >>> packed_s = mx.seg_recordio.pack(header, image_data, label_data)
    """
    # test_s = image_data + label_data
    # test_len = len(test_s)
    # image_len = len(image_data)
    # label_len = len(label_data)
    header = ISegRHeader(*header)
    s = struct.pack(_ISEGR_FORMAT, *header) + image_data + label_data
    # total_len = len(s)
    # if (image_len + label_len) != (header.image_size + header.label_size):
    #     print("{}<>{}+{}".format(total_len, header.image_size, header.label_size))
    return s

def unpack(s):
    """Unpack a MXImageRecord to string.

    Parameters
    ----------
    s : str
        String buffer from ``MXRecordIO.read``.

    Returns
    -------
    header : IRHeader
        Header of the image record.
    s : str
        Unpacked string.

    Examples
    --------
    >>> record = mx.seg_recordio.MXSegRecordIO('test.rec', 'r')
    >>> item = record.read()
    >>> header, s = mx.seg_recordio.unpack(item)
    >>> header
    HEADER(flag=0, label=0, image_len=368032, label_len=38742, id=20129312, id2=0)
    """
    header = ISegRHeader(*struct.unpack(_ISEGR_FORMAT, s[:_IR_SIZE]))
    s = s[_IR_SIZE:]
    if header.flag > 0:
        s = s[header.flag*4:]
    return header, s

def unpack_img(s, iscolor=-1):
    """Unpack a MXImageSegRecord to image.

    Parameters
    ----------
    s : str
        String buffer from ``MXSegRecordIO.read``.
    iscolor : int
        Image format option for ``cv2.imdecode``.

    Returns
    -------
    header : IRHeader
        Header of the image record.
    img : numpy.ndarray
        Unpacked image.

    Examples
    --------
    >>> record = mx.seg_recordio.MXSegRecordIO('test.rec', 'r')
    >>> item = record.read()
    >>> header, img, label = mx.seg_recordio.unpack_img(item)
    >>> header
    HEADER(flag=0, label=0, id=20129312, id2=0)
    >>> img
    array([[[ 23,  27,  45],
            [ 28,  32,  50],
            ...,
            [ 36,  40,  59],
            [ 35,  39,  58]],
           ...,
           [[ 91,  92, 113],
            [ 97,  98, 119],
            ...,
            [168, 169, 167],
            [166, 167, 165]]], dtype=uint8)
    """
    header, s = unpack(s)
    image_data = np.frombuffer(s, dtype=np.uint8, count=header.image_size, offset=0)
    label_data = np.frombuffer(s, dtype=np.uint8, count=header.label_size, offset=header.image_size)
    assert cv2 is not None
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    label = cv2.imdecode(label_data, cv2.IMREAD_GRAYSCALE)
    return header, image, label

def pack_img(header, img, label, quality=95, img_fmt='.jpg', label_fmt='.png'):
    """Pack an image into ``MXImageRecord``.

    Parameters
    ----------
    header : IRHeader
        Header of the image record.
    img : numpy.ndarray
        Image to be packed.
    label: numpy.ndarry
        Label to be packed
    quality : int
        Quality for JPEG encoding in range 1-100, or compression for PNG encoding in range 1-9.
    img_fmt : str
        Encoding of the image (.jpg for JPEG, .png for PNG).
    label_fmt : str
        Encoding of the label (.jpg for JPEG, .png for PNG).

    Returns
    -------
    s : str
        The packed string.

    Examples
    --------
    >>> id = 2574
    >>> image = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
    >>> ret, buf = cv2.imencode(".jpg", img)
    >>> assert ret, 'failed to encode image'
    >>> image_data = buf.tostring()
    >>> image_len = len(image_data)

    >>> label = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
    >>> ret, buf = cv2.imencode(".png", label)
    >>> assert ret, 'failed to encode label'
    >>> label_data = buf.tostring()
    >>> label_len = len(label_data)

    >>> header = mx.seg_recordio.ISegRHeader(0, 0, image_len, label_len, id, 0)
    >>> packed_s = mx.seg_recordio.pack_img(header, image, label)
    """
    assert cv2 is not None
    encode_params = None
    jpg_formats = ['.JPG', '.JPEG']
    png_formats = ['.PNG']
    if img_fmt.upper() in jpg_formats:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif img_fmt.upper() in png_formats:
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, quality]

    ret, buf = cv2.imencode(img_fmt, img, encode_params)
    assert ret, 'failed to encode image'
    image_data = buf.tostring()

    if label_fmt.upper() in jpg_formats:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif label_fmt.upper() in png_formats:
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, quality]
    ret, buf = cv2.imencode(label_fmt, label, encode_params)
    assert ret, 'failed to encode image'
    label_data = buf.tostring()

    image_len = len(image_data)
    label_len = len(label_data)
    header = ISegRHeader(header.flag, header.label, image_len, label_len, header.id, 0)

    return pack(header, image_data, label_data)
