import os
import numpy as np
from pathlib import Path

from ultralytics.engine import results


class DataWriter:
    def __init__(self) -> None:
        self.create_directories()
        self.imgpath = Path("new_data/images")
        self.labelpath = Path("new_data/labels")

        # get last file number

        self.cur_file = self.get_last_file()

    def create_directories(self) -> None:
        os.makedirs("new_data/images", exist_ok=True)
        os.makedirs("new_data/labels", exist_ok=True)

    # TODO: fix to allow resuming data collection. currently does not pick up from last file.
    def get_last_file(self) -> int:
        filenames = os.listdir(self.imgpath)
        if not filenames:
            return -1
        else:
            filenames = [int(fn.split(".")[0]) for fn in filenames]
            return max(filenames)

    def write_data(self, states):
        self.cur_file += 1
        state, tmp = states
        self.write_image(tmp)
        self.write_label(state)

    # TODO: currently has a bug that does not allow state, write, state, write again bc of seek file
    # maybe remove from states when written or something of the sort
    def write_image(self, tmpfile):
        """
        write an image in BGR format in a npy file
        """
        tmpfile.seek(0)
        fn = f"{self.cur_file:04d}.npy"
        path = self.imgpath / fn
        with open(path, "wb") as f:
            f.write(tmpfile.read())
        tmpfile.close()

    def write_label(self, results: results.Boxes):
        classes = results.cls.reshape(-1, 1)  # convert from (n, ) shape to (n, 1)
        xywh = results.xywhn  # shape of (n, 4)
        joined = np.concatenate((classes, xywh), axis=1)

        fn = f"{self.cur_file:04d}.txt"
        path = self.labelpath / fn

        with open(path, "w") as f:
            for row in joined:
                row = row.tolist()
                row[0] = int(row[0])
                line = " ".join(map(str, row)) + "\n"
                f.write(line)
