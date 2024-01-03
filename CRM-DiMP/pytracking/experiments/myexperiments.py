from pytracking.evaluation import Tracker, get_dataset, trackerlist


def atom_nfs_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('atom', 'default', range(3))

    dataset = get_dataset('nfs', 'uav')
    return trackers, dataset


def uav_test_xyl():
    n = 1
    trackers = trackerlist('dimp', 'super_dimp', range(n))

    dataset = get_dataset('mdot')
    return trackers, dataset
