'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat


if __name__ == "__main__":
    print("Run test...")
    test ="(0,25)	1064	174	2791	3373	269	2678	1902	3641	1216	915	3672	2803	2344	986	3217	2824	2598	464	2340	1952	1855	1353	1547	3487	3293	1541	2414	2728	340	1421	1963	2545	972	487	3463	2727	1135	3135	128	175	2423	1974	2515	3278	3079	1527	2182	1018	2800	1830	1539	617	247	3448	1699	1420	2487	198	811	1010	1423	2840	1770	881	1913	1803	1734	3326	1617	224	3352	1869	1182	1331	336	2517	1721	3512	3656	273	1026	1991	2190	998	3386	3369	185	2822	864	2854	3067	58	2551	2333	2688	3703	1300	1924	3118"
    tmp = test.split("\t")
    t2 = "(10,820)	1337	2889	1046	391	83	752	2699	1450	1433	1045	2478	3080	2284	3346	2376	3145	1951	2665	3441	2430	3056	3444	1227	829	1028	1610	3176	905	1767	2931	1859	986	1009	1805	1320	900	1786	2469	626	25	3532	2935	122	2551	2153	696	478	1871	2401	610	2938	508	1575	3510	2831	3316	2899	3557	2587	3319	2125	2134	1281	1082	2109	3701	2488	1301	1500	3285	3091	2147	1620	2592	3146	629	186	1977	1266	470	271	452	2865	1410	1608	1785	2780	392	954	3563	3364	733	1417	422	582	980	2771	424	2426"
    tmp2 = t2.split("\t")
    print(len(tmp[1:]))
    print(len(tmp2[1:]))

    t3 = "(209,2198)	1425	3639	958	2025	1942	2	2499	1435	1875	339	1707	2335	2271	1167	973	109	1502	1947	3539	2278	2925	3439	995	1577	666	569	1982	721	3086	689	106	494	2434	3203	3004	722	1944	685	2449	24	804	2841	296	2950	1557	648	3139	2980	1331	2398	2974	1422	2198	1163	765	280	859	1150	1247	3398	3199	3296	403	1597	600	3510	2762	2997	307	2187	3246	2251	485	1831	1433	1322	1399	603	797	673	3530	2820	1626	2223	2889	3568	1002	803	405	2701	1343	33	3172	2418	1224	3232	3453	3652	1923"
    tmp3 = t3.split("\t")
    print(tmp3[0], tmp3[1], tmp3[2], tmp3[3], tmp3[4])
    print(len(tmp3[1:]))




