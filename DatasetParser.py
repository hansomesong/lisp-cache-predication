from scapy.all import *
import ipaddress
import scipy.sparse as sp
import numpy as np
import os
from collections import defaultdict
from functools import wraps
import shelve

import time

def time_profiling(my_func):
    """
    credit: https://sanjass.github.io/notes/2019/07/28/timeit_decorator
    """
    @wraps(my_func)
    def timed(*args, **kw):
        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()
        print('"{}" took {:.3f} ms to execute\n'.format(my_func.__name__, (tend - tstart) * 1000))
        return output
    return timed


class DatasetParser(object):
    """
    This class is used to take a PCAP file as input, store the interaction information between source and destination IP
    address into a sparse matrix, and filly generate the training/test file for machine learning model.
    """

    @time_profiling
    def __init__(self, dataset_name, dataset_path, auto=True, prefix_mode=False, read_pcap_db=None):
        """
        @prefix_mode: flag to determine if network prefix should be parsed to be users and items.
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.auto = auto
        # one flag to determine population interaction with individual IP or IP prefix. Default is individual IP
        self.prefix_mode = prefix_mode
        # one flag to determine if we need to parse PCAP file
        self.read_pcap_db = read_pcap_db
        self.no_icmp = True
        self.no_dns = True
        self.no_nbms = True
        self.start_ip = ipaddress.ip_address(u"10.0.0.1")
        # Number of negative ID in negative test dataset
        self.test_negatives_num = 99
        # Number of IP address in interaction relationship with a given @IP
        self.interacted_ip_nb = 2
        # output file name
        file_base_name = os.path.splitext(os.path.basename(self.dataset_path))[0]


        self.dst_dir = "./Data"
        self.db_dst_dir = "./DB"
        self.trace_db_name = os.path.join(self.db_dst_dir, "{0}-{1}".format(self.dataset_name, file_base_name))
        self.training_test_dataset_name = "{0}-{1}.train-test.rating".format(self.dataset_name, file_base_name)
        self.training_dataset_name = "{0}-{1}.train.rating".format(self.dataset_name, file_base_name)
        self.test_dataset_name = "{0}-{1}.test.rating".format(self.dataset_name, file_base_name)
        self.test_negative_dataset_name = "{0}-{1}.test.negative".format(self.dataset_name, file_base_name)

        # call pcap_parser to count user/item number, populate interaction and interaction timestamp matrix
        if self.auto and self.read_pcap_db is None:
            # if on auto mode, interaction matrix and training dataset will be automatically generated.
            # other, the user has to manually call related method to populate matrix and generate dataset file.
            print("Dataset will be generated from traces file: {0}".format(self.dataset_path))
            self.pcap_parser()
            self.datasets_generator()
            # self.test_negative_dataset_generator()
        elif self.read_pcap_db:
            print("Dataset will be generated from traces related serialization file: {0}".format(self.read_pcap_db))
            self.trace_db_name = os.path.join(self.db_dst_dir, self.read_pcap_db)
            if os.path.exists(os.path.join(self.db_dst_dir, self.read_pcap_db)):
                self.object_deserialization(self.trace_db_name)
                self.datasets_generator()
            else:
                exit("Input serialization file {0} does not exist under {1}".format(self.read_pcap_db, self.db_dst_dir))
        else:
            exit("Error!")

    def get_network_addr(self0, ip_addr):
        """
         This utility method is used to retrieve the network prefix for a given network ip address.
         We assume that the network prefix mask is /24
        """
        tmp = ip_addr.split('.')
        tmp[3] = '0'
        return ".".join(tmp)


    def pcap_parser(self):
        """
        Note: 2020-08-25, by Qipeng
        We found that some IP addresses have only once interaction with other IP addresses. We decide to add some filter
        mechanism when populating sparse interaction matrix. For example, the IP address with less than 5 interactions
        won't be taken into consideration.

        Note: 2020-08-26, by Qipeng
        When parsing a trace file of 10 minutes long, pcap_parser takes almost 10 minutes to finish trace parsing.
        It's time to rethink our implementation. I think, we don't need to use sparse matrix here. Dict is sufficient.
        PS: sparse matrix is helpful when generating negative test file.

        Note: 2020-08-27, by Qipeng
        We need to consier seralize some objects so that we don't need to read and parse PCAP file each time.
        For PCAP file more than 6 minutes, the trace parsing is too much time-consuming!
        """

        scap_pcap = rdpcap(self.dataset_path)
        # for auckland-8 dataset, the starting IP address is 10.0.0.1
        src_ip_max = ipaddress.ip_address(u"10.0.0.1")
        dst_ip_max = ipaddress.ip_address(u"10.0.0.1")

        # first iteration of trace file: to count the number of source/destination IP address and initialize sparse matrix.
        # we use 'defaultdict' to store user(i.e. source IP address) and item(i.e. destination IP address).
        # If queried key does not exist, a list will be created.
        self.u_i_dict = defaultdict(list)
        for pkt in scap_pcap:
            if IP in pkt:
                if ICMP in pkt and self.no_icmp:
                    continue
                if UDP in pkt:
                    if pkt["UDP"].dport == 53 and self.no_dns:
                        continue
                    if pkt["UDP"].dport == 137 and self.no_nbms:
                        continue
                if not self.prefix_mode:
                    # host mode
                    ip_src = ipaddress.ip_address(unicode(pkt['IP'].src, "utf-8"))
                    ip_dst = ipaddress.ip_address(unicode(pkt['IP'].dst, "utf-8"))
                    # calculate user/item id from IP address. ID starts from 0.
                    user = int(ip_src) - int(self.start_ip)
                    item = int(ip_dst) - int(self.start_ip)
                else:
                    # prefix mode
                    # user and item in interaction matrix will be the network prefix, instead of particular ip address.
                    ip_src = ipaddress.ip_address(unicode(self.get_network_addr(pkt['IP'].src), "utf-8"))
                    ip_dst = ipaddress.ip_address(unicode(self.get_network_addr(pkt['IP'].dst), "utf-8"))
                    # the current implementation only consider the case of /24.
                    user = int(pkt['IP'].src.split('.')[2])
                    item = int(pkt['IP'].dst.split('.')[2])

                src_ip_max = max(src_ip_max, ip_src)
                dst_ip_max = max(dst_ip_max, ip_dst)
                self.u_i_dict[(user, item)].append(pkt.time)


        print(src_ip_max, dst_ip_max)
        self.u_i_dict = {x: [len(self.u_i_dict[x]), min(self.u_i_dict[x])] for x in self.u_i_dict.keys()}

        # To determine the shape of sparse matrix, we need to convert IPv4Address object to string
        # src_ip_max, dst_ip_max = str(src_ip_max), str(dst_ip_max)
        if self.prefix_mode:
            self.num_users = int(src_ip_max.split(".")[2])
            self.num_items = int(dst_ip_max.split(".")[2])

        else:
            # host mode
            self.num_users = int(src_ip_max) - int(self.start_ip) + 1
            self.num_items = int(dst_ip_max) - int(self.start_ip) + 1

        print("The sparse matrix's dimension will be: {0} X {1}".format(self.num_users, self.num_items))
        # matrix for the user-item interaction history
        # why we still need self.mat? because we need it to generate test negative file.
        self.mat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        # # natrix for the interaction timestamp
        # self.mat_time = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float64)
        for key, value in self.u_i_dict.items():
            self.mat[key] = value[0]

        # last step: serialize the generated trace related object to avoid parsing trace file in future
        print("Last step: serialization of trace related objects into {0}".format(self.trace_db_name))
        self.object_serialization(self.trace_db_name)


    def object_deserialization(self, db_name):
        db = shelve.open(db_name)
        self.u_i_dict = db['u_i_dict']
        self.mat = db['U_i_dok_mat']
        self.num_users = db['num_users']
        self.num_items = db['num_items']
        db.close()

    def object_serialization(self, db_name):
        db = shelve.open(db_name)
        db['u_i_dict'] = self.u_i_dict
        db['U_i_dok_mat'] = self.mat
        db['num_users'] = self.num_users
        db['num_items'] = self.num_items
        db.close()

    def datasets_generator(self):
        output_f_p = os.path.join(self.dst_dir, self.training_test_dataset_name)
        test_f_p = os.path.join(self.dst_dir, self.test_dataset_name)
        training_f_p = os.path.join(self.dst_dir, self.training_dataset_name)

        tmp_dict = defaultdict(list)
        for pair in sorted(self.u_i_dict.keys(), key=lambda x: (x[0], self.u_i_dict[x][1])):
            tmp_dict[pair[0]].append(pair[1])

        tmp_dict = dict(filter(lambda x: len(x[1]) > self.interacted_ip_nb, tmp_dict.items()))

        test_dict = {x: tmp_dict[x][-1] for x in tmp_dict.keys()}
        training_dict = {x: tmp_dict[x][:-1] for x in tmp_dict.keys()}


        print(test_dict)


        with open(output_f_p, 'w') as f_handler:
            for pair in sorted(self.u_i_dict.keys(), key=lambda x: (x[0], self.u_i_dict[x][1])):
                f_handler.write("{0:<5}\t{1:<5}\t{2:<5}\t{3:<15}\n".format(
                    pair[0], pair[1], self.u_i_dict[pair][0], self.u_i_dict[pair][1])
                )

        with open(training_f_p, 'w') as f_handler:
            f_handler.write("#{0}\t{1}\n".format(self.num_users, self.num_items)
            )
            for user_id in sorted(training_dict.keys()):
                for item_id in training_dict[user_id]:
                    u_i_pair = (user_id, item_id)
                    f_handler.write("{0:<5}\t{1:<5}\t{2:<5}\t{3:<15}\n".format(
                        user_id, item_id, self.u_i_dict[u_i_pair][0], self.u_i_dict[u_i_pair][1])
                    )

        # The generated dataset contains training and test data.
        # We need to split the obtained file into training dataset and test dataset
        with open(test_f_p, 'w') as f_handler:
            for user_id in test_dict.keys():
                item_id = test_dict[user_id]
                u_i_pair = (user_id, item_id)
                f_handler.write(
                    "{0:<5}\t {1:<5}\t{2:<5}\t{3:<15}\n".format(
                        user_id,
                        item_id,
                        self.u_i_dict[u_i_pair][0],
                        self.u_i_dict[u_i_pair][1]
                    )
                )

        negative_test_f_p = os.path.join(self.dst_dir, self.test_negative_dataset_name)
        # mat is of type sp.dok_matrix for sparse matrix.
        # Initialize a Dict object to store the negative instance matrix
        result_dict = {}
        user_num, item_num = self.mat.get_shape()
        for (u, i) in test_dict.items():
            # if current user_id is already in result_dict, go to next iteration
            if result_dict.has_key(u):
                continue
            else:
                # if current user_id is not in result_dict, populate the list of items without interaction.
                result_dict[u] = []
                # get negative instances from sparse matrix
                for t in xrange(self.test_negatives_num):
                    j = np.random.randint(item_num)
                    # train datatype = sp.dok_matrix. Method "has_key()" does not exist in current scipy
                    while self.mat.has_key((u, j)):
                        j = np.random.randint(item_num)
                    result_dict[u].append(j)
        # output the result_dict into a text file
        users = sorted(result_dict.keys())
        with open(negative_test_f_p, 'w') as f_handler:
            for user_id in users:
                neg_items = "\t".join([str(e) for e in result_dict[user_id]])
                # print("current user_id:", user_id, test_dict[user_id])
                f_handler.write("({0},{1})\t{2}\n".format(user_id, test_dict[user_id], neg_items))


if __name__ == "__main__":
    dataset_name = "auckland-8"
    data_set_path = "/Users/qsong/Downloads/15-1min.pcap"
    parser = DatasetParser(dataset_name, data_set_path, auto=True, read_pcap_db="auckland-8-15-1min.db")
    # print(parser.get_network_addr("10.0.4.125"))













