import os
import struct
import xml.etree.ElementTree as ET
from xml.dom import minidom
from dataset import Dataset, Sample
import errno
import tables
import numpy as np
from draw import draw_pts


def xml_gesture_to_sample(fopen, gname, subject_name, pid):
    tree = ET.parse(fopen)
    root = tree.getroot()
    trajectory = []
    time_s = []
    for child in root:
        x, y, t = int(child.attrib["X"]), int(child.attrib["Y"]), int(child.attrib["T"])
        trajectory.append([x, y])
        time_s.append(t)
    time_s = [time - time_s[0] for time in time_s]
    return Sample(
        sname=subject_name,
        gname=gname,
        ename=fopen,
        pid=pid,
        trajectory=np.array(trajectory),
        time_s=np.array(time_s)
    )


def write_sample_binary(fname, sample: Sample):
    """
    Write a sample to path fname in the binary format
    """
    with open(fname, "wb") as fout:
        pts = sample.trajectory
        time_s = sample.time_s
        m = 2

        #
        # Load global header
        #
        name = str(sample.gname).encode('utf-8')
        fout.write(struct.pack("<B", len(name)))
        fout.write(struct.pack("{}s".format(len(name)), name))
        fout.write(struct.pack("<I", 1))

        #
        # Load leader header
        #
        pt_cnt = len(pts)
        fout.write(struct.pack("<I", pt_cnt))  # pt_cnt
        fout.write(struct.pack("<I", m))  # component

        #
        # Load leader trajectory and timestamps
        #
        for idx in range(pt_cnt):
            fout.write(struct.pack("<f", pts[idx][0]))
            fout.write(struct.pack("<f", pts[idx][1]))

        fout.write(struct.pack("<I", pt_cnt))  # timestamps
        for idx in range(pt_cnt):
            fout.write(struct.pack("<f", time_s[idx]))


# Add XML header to output string
def prettify(elem):
    """
    Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def gds_to_bin():
    """
    Load and convert original GDS dataset to the format read by our dataset reader
    """

    opath = os.path.join(os.curdir, "..\\datasets\\xml_logs")  # Original Path
    tpath = os.path.join(os.curdir, "..\\datasets\\gds_bin\\training")  # Target Path

    print(f"Looking for gestures in {opath}")
    pid = -1
    for subject_name in next(os.walk(opath))[1]:
        pid += 1
        subject_path = os.path.join(opath, subject_name)
        seen_sample_names = {}

        for pace in next(os.walk(subject_path))[1]:

            pace_path = os.path.join(subject_path, pace)
            directory = os.fsencode(pace_path)

            for file in os.listdir(directory):

                filename = os.fsdecode(file)
                if filename.endswith('.xml'):
                    gname = filename[:-6]
                    if gname not in seen_sample_names:
                        seen_sample_names[gname] = 0
                    else:
                        seen_sample_names[gname] += 1

                    # read the xml sample
                    fopen = os.path.join(pace_path, filename)
                    sample = xml_gesture_to_sample(fopen, gname, subject_name, pid)

                    # write the binary sample
                    folder_path = f"{tpath}\\Sub_U0{subject_name[-2:]}\\{gname}"
                    fpath = os.path.join(folder_path, f"ex_{seen_sample_names[gname]}")

                    try:
                        os.makedirs(os.path.dirname(fpath))
                    except OSError as e:
                        if e.errno == errno.EEXIST:
                            pass
                        else:
                            raise

                    write_sample_binary(fpath, sample)


if __name__ == '__main__':
    gds_to_bin()

