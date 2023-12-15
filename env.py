import numpy as np
import networkx as nx
import gym
from gym import spaces
import copy
import wandb

class PlacingEnv(gym.Env):
    def __init__(   self,
                    netlists       = [
                        {
                            "M0": {"G": "DA", "S": "SS", "D": "DA"},
                            "M1": {"G": "DA", "S": "SS", "D": "DB"},
                            "M2": {"G": "DA", "S": "SS", "D": "DC"},
                            "M3": {"G": "DA", "S": "SS", "D": "DD"}
                        },
                        {
                            "M0": {"G": "GA", "S": "SASD", "D": "SBDA"},
                            "M1": {"G": "GBGC", "S": "SBDA", "D": "DB"},
                            "M2": {"G": "GBGC", "S": "SCDD", "D": "DC"},
                            "M3": {"G": "GD", "S": "SASD", "D": "SCDD"}
                        },
                        {
                            "M0": {"G": "GA", "S": "SA", "D": "SADBDCSD"},
                            "M1": {"G": "GB", "S": "SADBDCSD", "D": "DB"},
                            "M2": {"G": "GC", "S": "SADBDCSD", "D": "DC"},
                            "M3": {"G": "GD", "S": "SD", "D": "SADBDCSD"}
                        }
                    ],
                    M                      = np.array([4, 4, 8, 8]),
                    K                      = 1.3,
                    test_netlist_i         = 0,
                    general_train_mode     = False,
                    common_centroid        = True,
                    reward_mode            = 0,
                    config                 = {"gamma": 0.999, "batch_size": 2048, "learning_rate": 0.0003}
                ):

        super(PlacingEnv, self).__init__()

        # netlist & graph
        self.netlists               =   copy.deepcopy(netlists)
        self.netlist                =   copy.deepcopy(netlists[test_netlist_i])
        self.graph                  =   self.get_netlist_graph(self.netlist)

        # preprocessing : floorplanning & dummy add
        self.M                      =   copy.deepcopy(M)
        self.K                      =   K

        self.config                 =   config
        self.general_train          =   general_train_mode
        self.common_centroid        =   common_centroid

        # wandb
        if self.general_train:
            wandb.init(
                project="APRiL",
                name=f"{len(netlists[0])}_cc_{self.common_centroid}_{reward_mode}_cpu1",
                config=self.config
            )
        
        self.width, self.height, self.centroid\
                                    =   self.floorplanning(self.M, self.K)
        self.dummy_num              =   self.width*self.height - np.sum(self.M)
        self.initial_inst_nums      =   np.append(self.dummy_num, self.M)
        self.best_dispersion_degree =   self.calculate_best_dispersion_degree_val()
        self.best_lod               =   self.calculate_best_length_of_diffusion_val()
        self.edge_informations      =   self.get_edge_information()
        self.placing_order          =   self.get_placing_order_coordinates(self.width, self.height)

        # for during training
        self.inst_nums              =   copy.deepcopy(self.initial_inst_nums)
        self.instance_placing_grid  =   np.zeros((self.width, self.height), dtype=np.int32)
        self.cur_inst               =   0
        self.placed_inst_num        =   0
        
        self.placing_position       =   self.placing_order[0]


        self.db_generated = False
        self.total_db_num = np.zeros(self.height)
        self.lod                    =   np.zeros(len(self.M))
        self.prev_lod               =   np.zeros(len(self.M))
        self.centroid_dists         =   np.zeros(len(self.M))
        self.prev_centroid_dists    =   np.zeros(len(self.M))
        self.centroids              =   np.array([np.zeros(2) for _ in range(len(self.M))])
        self.prev_centroids         =   np.array([np.zeros(2) for _ in range(len(self.M))])
        self.dispersion_degrees     =   np.zeros(len(self.M))
        self.prev_dispersion_degrees=   np.zeros(len(self.M))

        # net based
        # self.routing_complexity     =   np.zeros(len(self.graph["edges"]))
        # self.prev_routing_complexity=   np.zeros(len(self.graph["edges"]))

        # inst based
        self.routing_complexity     =   np.zeros(len(self.M))
        self.prev_routing_complexity=   np.zeros(len(self.M))


        if reward_mode == 0:
            self.reward_coefficients    =   np.array([
                0,       # default
                -100,    # termination
                -1,      # diffusion break
                0.1,     # centroid dist
                0.04,    # dispersion degree
                0.02,    # LOD
                0.005    # routing complexity
            ], dtype=np.float32)
        if reward_mode == 1:
            self.reward_coefficients    =   np.array([
                0.5,                         # default
                -100,                        # termination
                0.3, 0.015,                  # centroid dist
                0.04, 0.002,                 # dispersion degree
                0.005, 0.00025,              # LOD
                0.001, 0.000001              # routing complexity
            ], dtype=np.float32)
        if reward_mode == 2:
            self.reward_coefficients    =   np.array([
                0.2,                         # default
                -100,                        # termination
                0.1, 0.005,                  # centroid dist
                0.04, 0.002,                 # dispersion degree
                0.005, 0.00025,              # LOD
                0.01, 0.00001                # routing complexity
            ], dtype=np.float32)
        self.total_reward = 0
        self.terminate = False
        self.all_placed = False

        # Define action and observation space
        # They must be gym.spaces objects
        # select instance and place or select Dummy
        self.action_space = spaces.Discrete(len(self.inst_nums))
        self.observation_space = spaces.Box(
                            # current placing position in ratio             (2)
                            # remaining instance order in ratio             (len(self.inst_nums))
                            # remaining inst nums                           (len(self.inst_nums))
                            # centroids                                     (len(self.M) * 2)
                            # edge informations for nearest 4 instances     (len(self.M) * 4)
                            # embedded edge information for all inst in row (len(self.M))
            low=-1, high=1, shape=(2+2*len(self.inst_nums)+7*len(self.M),), dtype=np.float32
        )


    def get_netlist_graph(self, netlist):

        graph = {"nodes": [], "edges": dict()}
        for key, value in netlist.items():
            for k, v in value.items():
                if v not in graph["nodes"]:
                    graph["nodes"].append(v)
                    graph["edges"][v] = []
                graph["edges"][v].append([key, k])
        # print(graph)
        return graph


    def get_edge_information(self):

        edge_infos = []

        for o in range(len(self.initial_inst_nums)):

            # dummy
            if o == 0:
                temp = np.ones(len(self.M), dtype=np.float32)

            # not dummy
            elif o != 0:
                net_d, net_s = list(self.netlist.values())[o-1]["D"], list(self.netlist.values())[o-1]["S"]
                inst_ds_pairs = self.graph["edges"][net_d] + self.graph["edges"][net_s]
                temp = -np.ones(len(self.M), dtype=np.float32)
                for inst in inst_ds_pairs:
                    if inst[1] != "G":
                        temp[list(self.netlist.keys()).index(inst[0])] += 1

            edge_infos.append(temp)

        return np.array(edge_infos, dtype=np.float32)


    def floorplanning(self, M, K):

        total_inst_num = np.sum(M)
        ideal_width = np.sqrt(total_inst_num*K)

        # width
        width = np.ceil(ideal_width) if np.ceil(ideal_width) % 2 == 0 else np.floor(ideal_width)

        # height
        height = np.ceil(total_inst_num/width)
        if np.any(self.M % 2 == 1) and height % 2 == 0:
            height += 1
            if (width-2)*height > np.sum(self.M): width -= 2
        if height < 3:
            height += 1
            if (width-2)*height > np.sum(self.M): width -= 2

        # centroid
        centroid = np.array([width/2-0.5, height/2-0.5], dtype=np.float32)

        return int(width), int(height), centroid


    def calculate_best_dispersion_degree_val(self):
        positions_x = np.array([]); positions_y = np.array([])
        for i in range(self.width):
            for j in range(self.height):
                positions_x = np.append(positions_x, i - self.centroid[0])
                positions_y = np.append(positions_y, j - self.centroid[1])
        
        dispersion_degree = np.sqrt(np.var(positions_x)+np.var(positions_y))

        return dispersion_degree


    def calculate_best_length_of_diffusion_val(self):
        best_lod_inv = 0
        for i in range(self.width):
            for j in range(self.height):
                best_lod_inv += 1/(i+1) + 1/(self.width-i)

        best_lod = 1/best_lod_inv
        best_lod /= np.sum(self.initial_inst_nums)

        return best_lod


    def get_placing_order_coordinates(self, width, height):

        orders = []

        if not self.common_centroid:
            for k in range(int(width/2 + np.ceil(height/2) - 1)):
                for j in range(k+1):
                    i = k-j
                    if i < width/2 and j < np.ceil(height/2):
                        orders.append([width/2 - 1 - i, np.ceil(height/2) - 1 - j])
                        if height%2 == 0 or j != 0:
                            orders.append([width/2 + i, height - np.ceil(height/2) + j])
                        orders.append([width/2 + i, np.ceil(height/2) - 1 - j])
                        if height%2 == 0 or j != 0:
                            orders.append([width/2 - 1 - i, height - np.ceil(height/2) + j])

        elif self.common_centroid:
            for k in range(int(width/2 + np.ceil(height/2) - 1)):
                for j in range(k+1):
                    i = k-j
                    if i < width/2 and j < np.ceil(height/2):
                        orders.append([width/2 - 1 - i, np.ceil(height/2) - 1 - j])
                        if height%2 == 0 or j != 0:
                            orders.append([width/2 + i, np.ceil(height/2) - 1 - j])

        return np.array(orders, dtype=np.int32)



    def get_obs(self):
        
        # Return Observation (normalize)
        base = self.placing_order[self.placed_inst_num]

        ######### current placement position
        cur_pos = base - self.centroid
        cur_pos[0] /= (self.width-1)/2
        cur_pos[1] /= (self.height-1)/2
        #########

        ######### remaining instances
        # remaining num
        remaining_inst_n = np.array(self.inst_nums, dtype=np.float32)\
                                /np.clip(np.array(self.initial_inst_nums, dtype=np.float32), 1, np.inf)
        remaining_inst_n = np.clip(remaining_inst_n, 0, 1)
        # msg_passed_remaining_inst_n = self.message_passing(remaining_inst_n)
        # remaining order
        remaining_order = np.array([], dtype=np.float32)
        prev_temp = np.zeros(1)
        for i, sorted_inst in enumerate(np.sort(self.inst_nums)):
            temp = sorted_inst
            if i == 0 or not np.all(temp == prev_temp):
                remaining_order = np.append(remaining_order, np.where(self.inst_nums == sorted_inst))
            prev_temp = sorted_inst
        remaining_order /= len(self.inst_nums)-1
        #########

        ######### centroids (average for normalization)
        centroids = np.zeros(2*len(self.M), dtype=np.float32)
        n_s = np.zeros(2*len(self.M), dtype=np.float32)
        for i in range(self.width):
            for j in range(self.height):
                inst = self.instance_placing_grid[i][j]
                if inst != 0:
                    centroids[2*inst-2] += (i - self.centroid[0])/(self.width-1)
                    centroids[2*inst-1] += (j - self.centroid[1])/(self.height-1)
                    n_s[2*inst-2] += 1
                    n_s[2*inst-1] += 1
        centroids /= np.clip(n_s, 1, np.inf)
        #########

        ######### nearest 4 instances
        # 1 if same instance, 0 if different instance but connected, -1 if no connection
        # include dummy as 0
        for x in range(self.width):
            o = self.instance_placing_grid[x][base[1]]
            if x == 0:
                os = np.reshape([self.edge_informations[o]], (1, len(self.M)))
            else:
                os = np.append(os, np.reshape(self.edge_informations[o], (1, len(self.M))), axis=0)
        edge_info_embed = np.mean(os, axis=0)
        o1 = self.instance_placing_grid[base[0]-2][base[1]] if base[0]-2 >= 0 else 0
        o2 = self.instance_placing_grid[base[0]-1][base[1]] if base[0]-1 >= 0 else 0
        o3 = self.instance_placing_grid[base[0]+1][base[1]] if base[0]+1 < self.width else 0
        o4 = self.instance_placing_grid[base[0]+2][base[1]] if base[0]+2 < self.width else 0
        o_s = [o1, o2, o3, o4]
        # getting edge informations for o1 ~ o4
        edge_infos = np.concatenate([self.edge_informations[o] for o in o_s], axis=0)
        #########

        obs = np.concatenate([  cur_pos,
                                remaining_inst_n, remaining_order,
                                centroids,
                                edge_info_embed,
                                edge_infos  ],
                            axis=0, dtype=np.float32)
        
        return obs



    def message_passing(self, remaining_inst_nums):
        
        drain_part = np.zeros(len(remaining_inst_nums), dtype=np.float32)
        source_part = np.zeros(len(remaining_inst_nums), dtype=np.float32)

        netlist_keys = list(self.netlist.keys())
        netlist_values = list(self.netlist.values())

        for i in range(len(remaining_inst_nums)):

            if i != 0:  # not dummy
                connection = netlist_values[i-1]
                drain_net, source_net = connection["D"], connection["S"]
                connected_instance_d = [netlist_keys.index(l[0])+1 for l in self.graph["edges"][drain_net]]
                connected_instance_s = [netlist_keys.index(l[0])+1 for l in self.graph["edges"][source_net]]

                drain_part[i] = 0.5*(np.mean([remaining_inst_nums[connected_instance_d]]) + remaining_inst_nums[i])
                source_part[i] = 0.5*(np.mean([remaining_inst_nums[connected_instance_s]]) + remaining_inst_nums[i])
            
            elif i == 0:    # dummy
                drain_part[i] = 0.5*(np.mean(remaining_inst_nums) + remaining_inst_nums[i])
                source_part[i] = 0.5*(np.mean(remaining_inst_nums) + remaining_inst_nums[i])
            

        return np.append(drain_part, source_part)



    def check_termination(self):
        # if the agent picks the instance which is all placed
        self.terminate = np.any(self.inst_nums < 0)
        self.all_placed = bool(np.all(self.placing_position == self.placing_order[-1]) and not self.terminate)


    def get_centroids_dists(self):
        # centroids
        self.centroids[self.cur_inst-1] = np.zeros(2)
        for j in range(self.height):
            for i in range(self.width):
                if self.cur_inst == self.instance_placing_grid[i][j]:
                    self.centroids[self.cur_inst-1] += (np.array([i, j])-self.centroid)
        self.centroids[self.cur_inst-1] /= np.count_nonzero(self.instance_placing_grid == self.cur_inst)
        # centroid dists
        self.centroid_dists = np.zeros(len(self.M))
        for i, c in enumerate(self.centroids):
            self.centroid_dists[i] = np.linalg.norm(c)


    def is_diffusion_break_generated(self):
        self.db_generated = False
        y = self.placing_position[1]
        self.total_db_num[y] = 0
        
        if self.placing_position[0] >= self.width/2:

            rds, rinst_lds = 0, 0

            for x in range(0,self.width-1):
                rds = -rinst_lds

                inst = self.instance_placing_grid[x][y]
                next_inst = self.instance_placing_grid[x+1][y]
                if inst*next_inst == 0: rds, rinst_lds = 0, 0; continue

                c_bool, c_info = self.connection_info(inst, next_inst)

                if c_bool:      # both connected instance
                    if rds==0:
                        # rds
                        for k in range(len(c_info)):
                            if c_info[k][0] != c_info[0][0]: break
                            if k == len(c_info)-1:
                                rinst_lds = c_info[0][1]

                    elif rds!=0:
                        # if lds/rds matches info
                        for k in range(len(c_info)):
                            if c_info[k][0] != c_info[0][0]:
                                if rds == c_info[0][0]:
                                    rinst_lds = c_info[0][1]
                                elif rds == c_info[k][0]:
                                    rinst_lds = c_info[k][1]
                                break
                            if k == len(c_info)-1:
                                if c_info[0][0] != rds:
                                    self.total_db_num[y] += 1
                                    if x+1 == self.placing_position[0]:
                                        self.db_generated = True; break
                                    rds, rinst_lds = 0, 0
                                else:
                                    rinst_lds = c_info[0][1]
                    
                else:
                    self.total_db_num[y] += 1
                    if x+1 == self.placing_position[0]:
                        self.db_generated = True; break
                    rds, rinst_lds = 0, 0
                    

        elif self.placing_position[0] < self.width/2:

            lds, linst_rds = 0, 0

            for x in range(self.width-1, 0, -1):
                lds = -linst_rds

                inst = self.instance_placing_grid[x][y]
                next_inst = self.instance_placing_grid[x-1][y]
                if inst*next_inst == 0: lds, linst_rds = 0, 0; continue

                c_bool, c_info = self.connection_info(inst, next_inst)

                if c_bool:      # both connected instance
                    if lds==0:
                        # rds
                        for k in range(len(c_info)):
                            if c_info[k][0] != c_info[0][0]: break
                            if k == len(c_info)-1:
                                linst_rds = c_info[0][1]

                    elif lds!=0:
                        # if lds/rds matches info
                        for k in range(len(c_info)):
                            if c_info[k][0] != c_info[0][0]:
                                if lds == c_info[0][0]:
                                    linst_rds = c_info[0][1]
                                elif lds == c_info[k][0]:
                                    linst_rds = c_info[k][1]
                                break
                            if k == len(c_info)-1:
                                if c_info[0][0] != lds:
                                    self.total_db_num[y] += 1
                                    if x-1 == self.placing_position[0]:
                                        self.db_generated = True; break
                                    lds, linst_rds = 0, 0
                                else:
                                    linst_rds = c_info[0][1]
                    
                else:
                    self.total_db_num[y] += 1
                    if x-1 == self.placing_position[0]:
                        self.db_generated = True; break
                    lds, linst_rds = 0, 0
        


    def get_lod(self):
        lod_inv = 0
        for j in range(self.height):
            for i in range(self.width):
                if self.cur_inst == self.instance_placing_grid[i][j]:
                    lod_inv += 1/(i+1) + 1/(self.width-i)
        self.lod[self.cur_inst-1] = 1/lod_inv
        self.lod[self.cur_inst-1] *= np.count_nonzero(self.instance_placing_grid == self.cur_inst)

    def get_dispersion_degrees(self):
        self.dispersion_degrees[self.cur_inst-1] = 0
        device_positions_x = []
        device_positions_y = []
        for j in range(self.height):
            for i in range(self.width):
                if self.cur_inst == self.instance_placing_grid[i][j]:
                    device_positions_x.append(i-self.centroid[0])
                    device_positions_y.append(j-self.centroid[1])
        self.dispersion_degrees[self.cur_inst-1] =\
            np.sqrt( np.var(device_positions_x) + np.var(device_positions_y) )

    # net based
    def get_routing_complexity_mst(self):
        
        netnames = list(self.graph["edges"].keys())
        net_graph = dict()
        for instance in self.netlist.keys():
            for gds in self.netlist[instance]:
                if self.netlist[instance][gds] not in net_graph.keys():
                    net_graph[self.netlist[instance][gds]] = []
                if instance not in net_graph[self.netlist[instance][gds]]:
                    net_graph[self.netlist[instance][gds]].append(instance)

        search_nets = []
        for e_k, e_v in net_graph.items():
            if list(self.netlist.keys())[self.cur_inst-1] in e_v:
                search_nets.append(e_k)
                self.routing_complexity[netnames.index(e_k)] = 0

        for netname in search_nets:
            indices = np.where(
                        np.isin(
                            self.instance_placing_grid,
                            [list(self.netlist.keys()).index(net_graph[netname][x])+1\
                                for x in range(len(net_graph[netname]))]
                        )
                    )
            row, column = indices

            # MST
            s = []
            t = []
            weights = []

            edge_count = 0
            for i in range(len(row) - 1):
                for j in range(i + 1, len(row)):
                    s.append(str(i))
                    t.append(str(j))
                    weights.append(abs(row[i] - row[j]) + abs(column[i] - column[j]))
                    edge_count += 1

            G = nx.Graph()
            for i in range(edge_count):
                G.add_edge(s[i], t[i], weight=weights[i])

            mst1 = nx.minimum_spanning_tree(G)
            wirelength = sum(mst1[i][j]["weight"] for i, j in mst1.edges)

            self.routing_complexity[netnames.index(netname)] = wirelength

    # inst based
    def get_routing_complexity(self):
        
        # find positions of currently placed instance
        indices = np.where( np.isin( self.instance_placing_grid, [inst for inst in range(1, len(self.M)+1)] ) )
        row, column = indices
        
        # maximum distances to the nearest connected instance
        critical_mindists = np.array([[0 for _ in range(len(self.M))]\
                                    if np.count_nonzero(self.instance_placing_grid == i+1) < 2\
                                    else [-1 for _ in range(len(self.M))]\
                                            for i in range(len(self.M))])
        # minimum distance of maximum distances
        minimum_maxdists = np.array([[0 for _ in range(len(self.M))]\
                                    if np.count_nonzero(self.instance_placing_grid == i+1) < 2\
                                    else [self.width+self.height for _ in range(len(self.M))]\
                                            for i in range(len(self.M))])

        for ind in range(len(row)):
            x, y = row[ind], column[ind]
            instance = self.instance_placing_grid[x][y]

            mindist = np.array([0 if np.count_nonzero(self.instance_placing_grid == i+1) < 2\
                             else self.width+self.height\
                                    for i in range(len(self.M))])
            maxdist = np.array([0 if np.count_nonzero(self.instance_placing_grid == i+1) < 2\
                             else -1\
                                    for i in range(len(self.M))])

            k = 1
            while np.any(mindist == self.width+self.height):
                for i in range(-k, k+1):
                    for j in [k-abs(i), -k+abs(i)]:
                        if x+i >=0 and x+i < self.width and y+j >= 0 and y+j < self.height:
                            inst = self.instance_placing_grid[x+i][y+j]
                        else: inst = 0
                        if inst != 0:
                            if self.connected(inst, instance):
                                mindist[inst-1] = np.min([mindist[inst-1], k])
                            else: mindist[inst-1] = 0
                k += 1
            critical_mindists[instance-1] = np.maximum(critical_mindists[instance-1], mindist)

            k = self.width+self.height-2
            while np.any(maxdist == -1):
                for i in range(-k, k+1):
                    for j in [k-abs(i), -k+abs(i)]:
                        if x+i >=0 and x+i < self.width and y+j >= 0 and y+j < self.height:
                            inst = self.instance_placing_grid[x+i][y+j]
                        else: inst = 0
                        if inst != 0:
                            if self.connected(inst, instance):
                                maxdist[inst-1] = np.max([maxdist[inst-1], k])
                            else: maxdist[inst-1] = 0
                k -= 1
            minimum_maxdists[instance-1] = np.minimum(minimum_maxdists[instance-1], maxdist)

        for rc_i in range(len(self.routing_complexity)):
            self.routing_complexity[rc_i] = np.sum(critical_mindists[rc_i])+np.sum(minimum_maxdists[rc_i])


    def recalculate_criterion_values(self):
        if self.cur_inst != 0:
            self.get_centroids_dists()
            self.get_dispersion_degrees()
            self.is_diffusion_break_generated()
            self.get_lod()
            self.get_routing_complexity()


    def get_rwd(self):
        r0, r1, r2, r3, r4, r5, r6 = 0, 0, 0, 0, 0, 0, 0

        # ===== Default ===== #
        r0 = self.reward_coefficients[0]
        # =================== #
        # ===== Termination ===== #
        if self.terminate:
            r1 = self.reward_coefficients[1]
        # ======================= #
        if self.cur_inst != 0:
            # ========== Diffusion Break =========== #
            if self.db_generated:
                r2 = self.reward_coefficients[2]
            # ====================================== #
            if self.prev_centroid_dists[self.cur_inst-1] != 0:
                # ===== Centroid Variation ===== #
                r3 = self.reward_coefficients[3]*(np.sum(self.prev_centroid_dists) - np.sum(self.centroid_dists))
                # ============================== #
            # ========= Dispersion Degrees ========= #
            r4 = self.reward_coefficients[4]*(  abs(self.best_dispersion_degree - self.prev_dispersion_degrees[self.cur_inst-1])\
                                                - abs(self.best_dispersion_degree - self.dispersion_degrees[self.cur_inst-1]) )
            # ====================================== #
            # ======== Length of Diffusion ========= #
            r5 = self.reward_coefficients[5]*(  abs(self.best_lod - self.prev_lod[self.cur_inst-1])\
                                                - abs(self.best_lod - self.lod[self.cur_inst-1])   )
            # ====================================== #
            # ========== Optimize Routing ========== #
            r6 = self.reward_coefficients[6]*( np.sum(self.prev_routing_complexity)\
                                            - np.sum(self.routing_complexity))
            # ====================================== #

        reward = r0 + r1 + r2 + r3 + r4 + r5 + r6

        # print()
        # print(f"default                     : {r0}")
        # print(f"termination                 : {r1}")
        # print(f"diffusion break             : {r2}")
        # print(f"centroids    : {self.centroids}")
        # print(f"centroid distance variation : {r3}")
        # print("\t", "current : ", self.centroid_dists, "\n\t",
        #             "prev    : ", self.prev_centroid_dists)
        # print(f"dispersion degree           : {r4}")
        # print("\t", "current : ", self.dispersion_degrees, "\n\t",
        #             "prev    : ", self.prev_dispersion_degrees, "\n\t",
        #             "best    : ", self.best_dispersion_degree)
        # print(f"length of diffusion         : {r5}")
        # print("\t", "current : ", self.lod, "\n\t",
        #             "prev    : ", self.prev_lod, "\n\t",
        #             "best    : ", self.best_lod)
        # print(f"routing complexity          : {r6}")
        # print("\t", "current : ", self.routing_complexity, "\n\t",
        #             "prev    : ", self.prev_routing_complexity)
        # print(f"reward       : {reward}")

        self.prev_lod = copy.deepcopy(self.lod)
        self.prev_dispersion_degrees = copy.deepcopy(self.dispersion_degrees)
        self.prev_routing_complexity = copy.deepcopy(self.routing_complexity)
        self.prev_centroids = copy.deepcopy(self.centroids)
        self.prev_centroid_dists = copy.deepcopy(self.centroid_dists)

        self.total_reward = self.config["gamma"]*self.total_reward + reward

        return float(reward)


    def connected(self, inst1, inst2):
        if not bool(inst1 != 0 and inst2 != 0): connected = True
        elif inst1 == inst2: connected = True
        else:         
            if set(list(self.netlist.values())[inst1-1]["S"]+list(self.netlist.values())[inst1-1]["D"])\
                & set(list(self.netlist.values())[inst2-1]["S"]+list(self.netlist.values())[inst2-1]["D"]) != None:
                    connected = True
            else: connected = False
        return connected


    def connection_info(self, inst1, inst2):
        if not bool(inst1 != 0 and inst2 != 0):
            connected = True
            info = [[-1, -1], [-1, 1], [1, -1], [1, 1]] # SS, SD, DS, DD
        elif inst1 == inst2:
            connected = True
            info = [[-1, -1], [1, 1]] # SS, DD
        else:
            inst1_S = list(self.netlist.values())[inst1-1]["S"]; inst1_D = list(self.netlist.values())[inst1-1]["D"]
            inst2_S = list(self.netlist.values())[inst2-1]["S"]; inst2_D = list(self.netlist.values())[inst2-1]["D"]
            if set((inst1_S, inst1_D)) & set((inst2_S, inst2_D)) != set():
                connected = True
                info = []
                if inst1_S == inst2_S: info.append([-1, -1])  # SS
                if inst1_S == inst2_D: info.append([-1, 1])  # SD
                if inst1_D == inst2_S: info.append([1, -1])  # DS
                if inst1_D == inst2_D: info.append([1, 1])  # DD
            else:
                connected = False; info = None
        return connected, info


    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """

        # preprocessing : floorplanning & dummy add
        if self.general_train:
            self.netlist                =   self.netlists[np.random.randint(0, len(self.netlists))]
            self.graph                  =   self.get_netlist_graph(self.netlist)
            self.M                      =   2*np.random.randint(1, 6, (len(self.netlist)), dtype=np.int32)
            self.K                      =   1.3 if np.random.randint(0, 2) == 0 else 2
            self.width, self.height, self.centroid\
                                        =   self.floorplanning(self.M, self.K)
            self.dummy_num              =   self.width*self.height - np.sum(self.M)
            self.initial_inst_nums      =   np.append(self.dummy_num, self.M)
            self.best_dispersion_degree =   self.calculate_best_dispersion_degree_val()
            self.best_lod               =   self.calculate_best_length_of_diffusion_val()
            self.edge_informations      =   self.get_edge_information()
            self.placing_order          =   self.get_placing_order_coordinates(self.width, self.height)

        # Initialize the agent
        self.inst_nums              =   copy.deepcopy(self.initial_inst_nums)
        self.instance_placing_grid  =   np.zeros((self.width, self.height), dtype=np.int32)
        self.cur_inst               =   0
        self.placed_inst_num        =   0
        
        self.placing_position       =   self.placing_order[0]
        
        self.db_generated           =   False
        self.total_db_num           =   np.zeros(self.height)
        self.lod                    =   np.zeros(len(self.M))
        self.prev_lod               =   np.zeros(len(self.M))
        self.centroid_dists         =   np.zeros(len(self.M))
        self.prev_centroid_dists    =   np.zeros(len(self.M))
        self.centroids              =   np.array([np.zeros(2) for _ in range(len(self.M))])
        self.prev_centroids         =   np.array([np.zeros(2) for _ in range(len(self.M))])
        self.dispersion_degrees     =   np.zeros(len(self.M))
        self.prev_dispersion_degrees=   np.zeros(len(self.M))
        # net based
        # self.routing_complexity     =   np.zeros(len(self.graph["edges"]))
        # self.prev_routing_complexity=   np.zeros(len(self.graph["edges"]))

        # inst based
        self.routing_complexity     =   np.zeros(len(self.M))
        self.prev_routing_complexity=   np.zeros(len(self.M))

        self.total_reward = 0
        self.terminate = False
        self.all_placed = False

        obs = self.get_obs()
        
        return obs  # empty info dict



    def step(self, action):
        # action == 0:   dummy
        # action == 1:   M1
        # action == 2:   M2
        # action == 3:   M3
        # action == 4:   M4
        # .....

        self.cur_inst = action

        # observation
        # placement grid
        if self.placed_inst_num < len(self.placing_order):
            # placing position
            self.placing_position = self.placing_order[self.placed_inst_num]
            self.instance_placing_grid[self.placing_position[0]]\
                                      [self.placing_position[1]] = self.cur_inst
            self.inst_nums[self.cur_inst] -= 1

            # mirroring when common centroid mode
            if self.common_centroid:
                self.instance_placing_grid[self.width - 1 - self.placing_position[0]]\
                                          [self.height - 1 - self.placing_position[1]] = self.cur_inst
                self.inst_nums[self.cur_inst] -= 1

        # place inst num
        if self.placed_inst_num < len(self.placing_order)-1:
            self.placed_inst_num += 1

        # termination condition
        self.recalculate_criterion_values()
        self.check_termination()

        # REWARD
        reward = self.get_rwd()

        # observation
        obs = self.get_obs()

        # done
        done = bool(self.terminate or self.all_placed)

        # Optionally we can pass additional info
        info = {
                    "pattern": np.transpose(self.instance_placing_grid),
                    "reward": self.total_reward,
                    "centroid_dists": self.centroid_dists,
                    "dispersion_degrees": self.dispersion_degrees,
                    "diffusion_break_num": self.total_db_num,
                    "LOD": self.lod,
                    "routing_complexity": self.routing_complexity
                }

        if self.general_train and done:
            wandb.log({
                "reward": self.total_reward,
                "centroid_dists_sum": np.sum(self.centroid_dists),
                "dispersion_degrees_var": np.var(self.dispersion_degrees),
                "diffusion_break": np.sum(self.total_db_num),
                "LOD_var": np.var(self.lod),
                "routing_complexity_sum": np.sum(self.routing_complexity)
            })

        return obs, reward, done, info



    def render(self):

        from colorama import init as colorama_init
        from colorama import Fore
        from colorama import Style
        colorama_init()

        print()
        print(f"initial instance num : \n{['dummy']+list(self.netlist.keys())}\n{np.append(self.dummy_num, self.M)}")        

        rendering_grid = []
        for j in range(self.height):
            row_els = []
            for i in range(self.width):
                inst = self.instance_placing_grid[i][j]
                if inst == 0:
                    el = "0"
                else:
                    el = f"{list(self.netlist.keys())[inst-1]}"
                row_els.append(el)
            rendering_grid.append(row_els)

        print("===========================================================================================")
        print(f"placed instance num         : {np.sum(self.initial_inst_nums) - np.sum(self.inst_nums)}/{self.width*self.height}")
        print(f"current inst nums           : {self.inst_nums}")
        print(f"current placements : \n{np.transpose(self.instance_placing_grid)}")
        print(f"{Fore.BLUE}", end="")
        print(f"diffusion break num         : {Style.BRIGHT}{self.total_db_num} -> {np.sum(self.total_db_num)}{Style.NORMAL}")
        print(f"centroid distances          : {self.centroid_dists}")
        print(f"centroid distances sum      : {Style.BRIGHT}{np.sum(self.centroid_dists)}{Style.NORMAL}")
        print(f"mean LOD values             : {self.lod}")
        print(f"mean LOD variance           : {Style.BRIGHT}{np.var(self.lod)}{Style.NORMAL}")
        print(f"dispersion degrees          : {self.dispersion_degrees}")
        print(f"dispersion degree var       : {Style.BRIGHT}{np.var(self.dispersion_degrees)}{Style.NORMAL}")
        print(f"routing complexity          : {self.routing_complexity}")
        print(f"route complexity sum        : {Style.BRIGHT}{np.sum(self.routing_complexity)}{Style.RESET_ALL}")
        print()

        print(f"{Fore.GREEN}{Style.NORMAL}", end="")
        for j in range(self.height):
            for i in range(self.width):
                print(rendering_grid[j][i], end=" ")
            print()
        print(f"{Style.RESET_ALL}", end="")
        print()
        print(f"total reward : {self.total_reward:.2f}")

        # print(f"easier placement grid :")
        # for row in rendering_grid:
        #     print(row)
        print("===========================================================================================")
        print()
        
    def close(self):
        pass




from stable_baselines3.common.env_checker import check_env
if __name__ == "__main__":
    place_env = PlacingEnv(
                            netlists       =   [
                                {
                                    "A": {"G": "DA", "S": "SS", "D": "DA"},
                                    "B": {"G": "DA", "S": "SS", "D": "DB"},
                                    "C": {"G": "DA", "S": "SS", "D": "DC"},
                                    "D": {"G": "DA", "S": "SS", "D": "DD"}
                                },
                                {
                                    "A": {"G": "GA", "S": "SADC", "D": "DA"},
                                    "B": {"G": "GB", "S": "SBDD", "D": "DB"},
                                    "C": {"G": "GCD", "S": "SCD", "D": "SADC"},
                                    "D": {"G": "GCD", "S": "SCD", "D": "SBDD"}
                                },
                                {
                                    "A": {"G": "GA", "S": "SA", "D": "SADBDCSD"},
                                    "B": {"G": "GB", "S": "SADBDCSD", "D": "DB"},
                                    "C": {"G": "GC", "S": "SADBDCSD", "D": "DC"},
                                    "D": {"G": "GD", "S": "SD", "D": "SADBDCSD"}
                                }
                            ],
                            test_netlist_i = 0,
                            M              = np.array([6, 6, 10, 10]),
                            K              = 2,
                            general_train_mode     = False,
                            common_centroid        = False,
                            reward_mode            = 0)
    # If the environment don't follow the interface, an error will be thrown
    check_env(place_env, warn=True)

    obs = place_env.reset()
    done = False

    # action_ex_list = [1, 2, 3, 4, 3, 4, 3, 4, 2, 1, 4, 3, 3, 4, 1, 2]
    # action_ex_list = [1, 2, 2, 1, 3, 3, 4, 4, 4, 4, 3, 3]
    action_ex_list = [1, 1, 3, 3, 1, 1, 3, 3, 4, 4, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 2, 2, 4, 4]

    k = 0
    while not done and k < len(action_ex_list):
        action = action_ex_list[k]; k+=1
        obs, reward, done, info = place_env.step(action)
        # print(reward)
        place_env.render()