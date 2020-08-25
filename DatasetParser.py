from scapy.all import *
import ipaddress
import scipy.sparse as sp
import numpy as np
import os
from collections import defaultdict

class DatasetParser(object):
    """
    This class is used to take a PCAP file as input, store the interaction information between source and destination IP
    address into a sparse matrix, and filly generate the training/test file for machine learning model.
    """

    def __init__(self, dataset_path, auto=True, prefix_mode=False):
        """
        @prefix_mode: flag to determine if network prefix should be parsed to be users and items.
        """
        self.dataset_path = dataset_path
        self.auto = auto
        # one flag to determine population interaction with individual IP or IP prefix. Default is individual IP
        self.prefix_mode = prefix_mode
        self.no_icmp = True
        self.no_dns = True
        self.no_nbms = True
        self.start_ip = ipaddress.ip_address(u"10.0.0.1")
        # Number of negative ID in negative test dataset
        self.test_negatives_num = 99
        self.training_test_dataset_name = "auckland-8.train-test.rating"
        self.training_dataset_name = "auckland-8.train.rating"
        self.test_dataset_name = "auckland-8.test.rating"
        self.test_negative_dataset_name = "auckland-8.test.negative"


        self.dst_dir = "./Data"

        # call pcap_parser to count user/item number, populate interaction and interaction timestamp matrix
        if self.auto:
            # if on auto mode, interaction matrix and training dataset will be automatically generated.
            # other, the user has to manually call related method to populate matrix and generate dataset file.
            self.pcap_parser()
            self.training_test_dataset_generator()
            # self.test_negative_dataset_generator()

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
        """
        scap_pcap = rdpcap(self.dataset_path)
        # for auckland-8 dataset, the starting IP address is 10.0.0.1
        src_ip_max = ipaddress.ip_address(u"10.0.0.1")
        dst_ip_max = ipaddress.ip_address(u"10.0.0.1")

        # first iteration of trace file: to count the number of source/destination IP address and initialize sparse matrix.
        u_i_pairs = []
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
                    user = int(pkt['IP'].src.split('.')[2]) + int(pkt['IP'].src.split('.')[3])
                    item = int(pkt['IP'].dst.split('.')[2]) + int(pkt['IP'].dst.split('.')[3])
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
                u_i_pairs.append((user, item))

        print(src_ip_max, dst_ip_max)

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
        self.mat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        # natrix for the interaction timestamp
        self.mat_time = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float64)


        # second iteration of trace:
        # to populate the interaction matrix (a sparse matrix) between source and destination IP address, along side
        # the packet time.
        for pkt in scap_pcap:
            if IP in pkt:
                # filter out ICMP packet
                if ICMP in pkt and self.no_icmp:
                    continue
                # filter DNS and NBMS packet
                if UDP in pkt:
                    if pkt["UDP"].dport == 53 and self.no_dns:
                        continue
                    if pkt["UDP"].dport == 137 and self.no_nbms:
                        continue
                ip_src = ipaddress.ip_address(unicode(pkt['IP'].src, "utf-8"))
                ip_dst = ipaddress.ip_address(unicode(pkt['IP'].dst, "utf-8"))
                # We would like that the user id start from 0
                user = int(ip_src) - int(self.start_ip)
                item = int(ip_dst) - int(self.start_ip)
                # print(user, item)
                self.mat[user, item] += 1.0
                if self.mat_time[user, item] == 0.0:
                    self.mat_time[user, item] = pkt.time
                # else:
                #     mat_time[user, item] = min(pkt.time, mat_time[user, item])

    def training_test_dataset_generator(self):
        output_f_p = os.path.join(self.dst_dir, self.training_test_dataset_name)
        test_f_p = os.path.join(self.dst_dir, self.test_dataset_name)
        training_f_p = os.path.join(self.dst_dir, self.training_dataset_name)


        tmp_dict = defaultdict(list)
        for pair in sorted(self.mat.keys(), key=lambda x: (x[0], self.mat_time[x])):
            tmp_dict[pair[0]].append(pair[1])

        tmp_dict = dict(filter(lambda x: len(x[1]) > 5, tmp_dict.items()))

        test_dict = {x: tmp_dict[x][-1] for x in tmp_dict.keys()}
        training_dict = {x: tmp_dict[x][:-1] for x in tmp_dict.keys()}


        print(test_dict)


        with open(output_f_p, 'w') as f_handler:
            for pair in sorted(self.mat.keys(), key=lambda x: (x[0], self.mat_time[x])):
                f_handler.write("{0:<5}\t{1:<5}\t{2:<5}\t{3:<15}\n".format(pair[0], pair[1], self.mat[pair], self.mat_time[pair]))

        with open(training_f_p, 'w') as f_handler:
            for user_id in sorted(training_dict.keys()):
                for item_id in training_dict[user_id]:
                    u_i_pair = (user_id, item_id)
                    f_handler.write("{0:<5}\t{1:<5}\t{2:<5}\t{3:<15}\n".format(
                        user_id, item_id, self.mat[u_i_pair], self.mat_time[u_i_pair])
                    )

        # The generated dataset contains training and test data.
        # We need to split the obtained file into training dataset and test dataset
        with open(test_f_p, 'w') as f_handler:
            for user_id in test_dict.keys():
                item_id = test_dict[user_id]
                f_handler.write(
                    "{0:<5}\t {1:<5}\t{2:<5}\t{3:<15}\n".format(
                        user_id,
                        item_id,
                        self.mat[(user_id, item_id)],
                        self.mat_time[(user_id, item_id)]
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
                print("current user_id:", user_id, test_dict[user_id])
                f_handler.write("({0},{1})\t{2}\n".format(user_id, test_dict[user_id], neg_items))


    def test_negative_dataset_generator(self):
        output_f_p = os.path.join(self.dst_dir, self.test_negative_dataset_name)
        # mat is of type sp.dok_matrix for sparse matrix.
        # Initialize a Dict object to store the negative instance matrix
        result_dict = {}
        user_num, item_num = self.mat.get_shape()
        for (u, i) in self.mat.keys():
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
        with open(output_f_p, 'w') as f_handler:
            for user in users:
                neg_items = " ".join([str(e) for e in result_dict[user]])
                f_handler.write("{0:<15}\t\{1}\n".format(user, neg_items))






if __name__ == "__main__":
    data_set_path = "/Users/qsong/Downloads/10min.pcap"
    parser = DatasetParser(data_set_path, auto=True)
    print(parser.get_network_addr("10.0.4.125"))













