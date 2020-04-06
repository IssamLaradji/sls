import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from sklearn import metrics
from haven import haven_utils as hu


def get_dataset(dataset_name, train_flag, datadir, exp_dict):
    if dataset_name == "mnist":
        dataset = torchvision.datasets.MNIST(datadir, train=train_flag,
                               download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.5,), (0.5,))
                               ]))

    if dataset_name == "cifar10":
        transform_function = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        dataset = torchvision.datasets.CIFAR10(
            root=datadir,
            train=train_flag,
            download=True,
            transform=transform_function)

    if dataset_name == "cifar100":
        transform_function = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        dataset = torchvision.datasets.CIFAR100(
            root=datadir,
            train=train_flag,
            download=True,
            transform=transform_function)

    if dataset_name in ["mushrooms", "w8a",
                        "rcv1", "ijcnn"]:

        sigma_dict = {"mushrooms": 0.5,
                      "w8a":20.0,
                      "rcv1":0.25 ,
                      "ijcnn":0.05}

        X, y = load_libsvm(dataset_name, data_dir=datadir)

        labels = np.unique(y)

        y[y==labels[0]] = 0
        y[y==labels[1]] = 1
		# TODO: (amishkin) splits = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=9513451)
        splits = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
        X_train, X_test, Y_train, Y_test = splits


        if train_flag:
            # fname_rbf = "%s/rbf_%s_train.pkl" % (datadir, dataset_name)

            # if os.path.exists(fname_rbf):
            #     k_train_X = hu.load_pkl(fname_rbf)
            # else:
            k_train_X = rbf_kernel(X_train, X_train, sigma_dict[dataset_name])
                # hu.save_pkl(fname_rbf, k_train_X)

            X_train = k_train_X
            X_train = torch.FloatTensor(X_train)
            Y_train = torch.FloatTensor(Y_train)

            dataset = torch.utils.data.TensorDataset(X_train, Y_train)

        else:
            # fname_rbf = "%s/rbf_%s_test.pkl" % (datadir, dataset_name)
            # if os.path.exists(fname_rbf):
            #     k_test_X = hu.load_pkl(fname_rbf)
            # else:
            k_test_X = rbf_kernel(X_test, X_train, sigma_dict[dataset_name])
                # hu.save_pkl(fname_rbf, k_test_X)

            X_test = k_test_X
            X_test = torch.FloatTensor(X_test)
            Y_test = torch.FloatTensor(Y_test)

            dataset = torch.utils.data.TensorDataset(X_test, Y_test)

        return dataset

    if dataset_name == "synthetic":
        margin = exp_dict["margin"]

        X, y, _, _ = make_binary_linear(n=exp_dict["n_samples"],
                                        d=exp_dict["d"],
                                        margin=margin,
                                        y01=True,
                                        bias=True,
                                        separable=True,
                                        seed=42)
        # No shuffling to keep the support vectors inside the training set
        splits = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
        X_train, X_test, Y_train, Y_test = splits

        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)

        Y_train = torch.FloatTensor(Y_train)
        Y_test = torch.FloatTensor(Y_test)

        if train_flag:
            dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        else:
            dataset = torch.utils.data.TensorDataset(X_test, Y_test)

        return dataset

    if dataset_name == "matrix_fac":
        fname = datadir + 'matrix_fac.pkl'
        if not os.path.exists(fname):
            data = generate_synthetic_matrix_factorization_data()
            hu.save_pkl(fname, data)

        A, y = hu.load_pkl(fname)

        X_train, X_test, y_train, y_test = train_test_split(A, y, test_size=0.2, random_state=9513451)

        training_set = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
        test_set = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))

        if train_flag:
            dataset = training_set
        else:
            dataset = test_set

    return dataset


# ===========================================================
# matrix matrix_factorization
def generate_synthetic_matrix_factorization_data(xdim=6, ydim=10, nsamples=1000, A_condition_number=1e-10):
	"""
    Generate a synthetic matrix factorization dataset as suggested by Ben Recht.
		See: https://github.com/benjamin-recht/shallow-linear-net/blob/master/TwoLayerLinearNets.ipynb.
	"""
	Atrue = np.linspace(1, A_condition_number, ydim
					   ).reshape(-1, 1) * np.random.rand(ydim, xdim)
	# the inputs
	X = np.random.randn(xdim, nsamples)
	# the y's to fit
	Ytrue = Atrue.dot(X)
	data = (X.T, Ytrue.T)

	return data

# ===========================================================
# Helpers
import os
import urllib

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from torchvision.datasets import MNIST


LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
LIBSVM_DOWNLOAD_FN = {"rcv1"       : "rcv1_train.binary.bz2",
                      "mushrooms"  : "mushrooms",
                      "ijcnn"      : "ijcnn1.tr.bz2",
                      "w8a"        : "w8a"}


def load_mnist(data_dir):
    dataset = MNIST(data_dir, train=True, transform=None,
          target_transform=None, download=True)

    X, y = dataset.data.numpy(), dataset.targets.numpy()
    X = X / 255.
    X = X.reshape((X.shape[0], -1))
    return X, y


def load_libsvm(name, data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    fn = LIBSVM_DOWNLOAD_FN[name]
    data_path = os.path.join(data_dir, fn)

    if not os.path.exists(data_path):
        url = urllib.parse.urljoin(LIBSVM_URL, fn)
        print("Downloading from %s" % url)
        urllib.request.urlretrieve(url, data_path)
        print("Download complete.")

    X, y = load_svmlight_file(data_path)
    return X, y


def make_binary_linear(n, d, margin, y01=False, bias=False, separable=True, shuffle=True, seed=None):
    assert margin >= 0.

    if seed:
        np.random.seed(seed)

    labels = [-1, 1]

    # Generate support vectors that are 2 margins away from each other
    # that is also linearly separable by a homogeneous separator
    w = np.random.randn(d); w /= np.linalg.norm(w)
    # Now we have the normal vector of the separating hyperplane, generate
    # a random point on this plane, which should be orthogonal to w
    p = np.random.randn(d-1); l = (-p@w[:d-1])/w[-1]
    p = np.append(p, [l])

    # Now we take p as the starting point and move along the direction of w
    # by m and -m to obtain our support vectors
    v0 = p - margin*w
    v1 = p + margin*w
    yv = np.copy(labels)

    # Start generating points with rejection sampling
    X = []; y = []
    for i in range(n-2):
        label = np.random.choice(labels)
        # Generate a random point with mean at the center with variance
        # adapted to the margin
        xi = np.sqrt(margin)*np.random.randn(d)
        dist = xi@w
        while dist*label <= margin:
            u = v0-v1 if label == -1 else v1-v0
            u /= np.linalg.norm(u)
            xi = xi + np.sqrt(margin)*u
            dist = xi@w

        X.append(xi)
        y.append(label)

    X = np.array(X).astype(float); y = np.array(y).astype(float)

    if shuffle:
        ind = np.random.permutation(n-2)
        X = X[ind]; y = y[ind]

    # Put the support vectors at the beginning
    X = np.r_[np.array([v0, v1]), X]
    y = np.r_[np.array(yv), y]

    if separable:
        # Assert linear separability
        # Since we're supposed to interpolate, we should not regularize.
        clff = SVC(kernel="linear", gamma="auto", tol=1e-10, C=1e10)
        clff.fit(X, y)
        assert clff.score(X, y) == 1.0

        # Assert margin obtained is what we asked for
        w = clff.coef_.flatten()
        sv_margin = np.min(np.abs(clff.decision_function(X)/np.linalg.norm(w)))
        assert np.abs(sv_margin - margin) < 1e-4
    else:
        flip_ind = np.random.choice(n, int(n*0.01))
        y[flip_ind] = -y[flip_ind]

    if y01:
        y[y==-1] = 0

    if bias:
        # TODO: Get rid of this later, bias should be handled internally,
        #       this is just for ease of implementation for the Hessian
        X = np.c_[np.ones(n), X]

    return X, y, w, (v0, v1)

def rbf_kernel(A, B, sigma):
    distsq = np.square(metrics.pairwise.pairwise_distances(A, B, metric="euclidean"))
    K = np.exp(-1 * distsq/(2*sigma**2))
    return K