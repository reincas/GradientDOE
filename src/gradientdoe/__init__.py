##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import zipfile

__version__ = "0.3.0"


class RawItem:
    """ File-like class mapping a byte sequence inside a given data container item. """

    def __init__(self, filename, itemname):
        with zipfile.ZipFile(filename, 'r') as z:
            info = z.getinfo(itemname)

            # Item must be uncompressed
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"Uncompressed item {itemname} required!")

            # Get offset of HDF5 file content
            with open(filename, 'rb') as f:
                f.seek(info.header_offset)

                # Extract length of filename and extra field from file header (ZIP)
                header_data = f.read(30)
                fn_len = int.from_bytes(header_data[26:28], 'little')
                extra_len = int.from_bytes(header_data[28:30], 'little')

                # Byte offset of the actual HDF5 file data
                self.offset = info.header_offset + 30 + fn_len + extra_len
                self.size = info.file_size

        self.f = open(filename, 'rb')
        self.f.seek(self.offset)

    def read(self, n=-1):
        if n == -1 or self.tell() + n > self.size:
            n = self.size - self.tell()
        return self.f.read(n)

    def seek(self, n, whence=0):
        if whence == 0:
            self.f.seek(self.offset + n)
        elif whence == 1:
            self.f.seek(n, 1)
        elif whence == 2:
            self.f.seek(self.offset + self.size + n)

    def tell(self):
        return self.f.tell() - self.offset

    def close(self):
        self.f.close()
