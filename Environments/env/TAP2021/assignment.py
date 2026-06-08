import math
import multiprocessing
import time
import heapq
import networkx as nx
from scipy.optimize import fsolve
import warnings
from Environments.env.TAP2021.network_import import *
from Environments.env.TAP2021.utils import PathUtils

warnings.filterwarnings('ignore', 'The iteration is not making good progress')
do_mp = True

class FlowTransportNetwork:

    def __init__(self):
        self.linkSet = {}
        self.nodeSet = {}

        self.tripSet = {}
        self.zoneSet = {}
        self.originZones = {}
        self.cost: float = np.nan

        self.networkx_graph = None

    def to_networkx(self, force=False):
        if force is True:
            del self.networkx_graph
            self.networkx_graph = nx.DiGraph([(int(begin), int(end)) for (begin, end) in self.linkSet.keys()])
            return self.networkx_graph
        elif self.networkx_graph is None:
            self.networkx_graph = nx.DiGraph([(int(begin),int(end)) for (begin,end) in self.linkSet.keys()])
        return self.networkx_graph

    def reset_flow(self):
        for link in self.linkSet.values():
            link.reset_flow()

    def reset(self):
        for link in self.linkSet.values():
            link.reset()


class Zone:
    def __init__(self, zoneId: str):
        self.zoneId = zoneId

        self.lat = 0
        self.lon = 0
        self.destList = []  # list of zone ids (strs)


class Node:
    """
    This class has attributes associated with any node
    """

    def __init__(self, nodeId: str):
        self.Id = nodeId

        self.lat = 0
        self.lon = 0

        self.outLinks = []  # list of node ids (strs)
        self.inLinks = []  # list of node ids (strs)

        # For Dijkstra
        self.label = np.inf
        self.pred = None


class Link:
    """
    This class has attributes associated with any link
    """

    def __init__(self,
                 init_node: str,
                 term_node: str,
                 capacity: float,
                 length: float,
                 fft: float,
                 b: float,
                 power: float,
                 speed_limit: float,
                 toll: float,
                 linkType
                 ):
        self.init_node = init_node
        self.term_node = term_node
        self.max_capacity = float(capacity)  # veh per hour
        self.length = float(length)  # Length
        self.normal_fft = float(fft)
        self.fft = float(fft)  # Free flow travel time (min)
        self.beta = float(power)
        self.alpha = float(b)
        self.speedLimit = float(speed_limit)
        self.toll = float(toll)
        self.linkType = linkType

        self.curr_capacity_percentage = 1
        self.capacity = self.max_capacity
        self.flow = 0.0
        self.cost = self.fft

    # Method not used for assignment
    def modify_capacity(self, delta_percentage: float = 0, set_capacity: float = -1) -> None:
        if set_capacity != -1:
            self.capacity = set_capacity
            self.curr_capacity_percentage = self.capacity / self.max_capacity
        else:
            assert -1 <= delta_percentage <= 1
            self.curr_capacity_percentage += delta_percentage
            self.curr_capacity_percentage = np.clip(self.curr_capacity_percentage, 0, 1)
            self.capacity = self.max_capacity * self.curr_capacity_percentage

    def modify_fft(self, modification):
        self.fft = self.fft * modification

    def reset(self):
        self.curr_capacity_percentage = 1
        self.capacity = self.max_capacity
        try:
            self.fft = self.normal_fft # TODO: Remove at some point
        except:
            pass
        self.reset_flow()

    def reset_flow(self):
        self.flow = 0.0
        self.cost = self.fft


class Demand:
    def __init__(self,
                 init_node: str,
                 term_node: str,
                 demand: float
                 ):
        self.fromZone = init_node
        self.toNode = term_node
        self.demand = float(demand)


def DijkstraHeap(origin, network: FlowTransportNetwork):
    """
    Calculates shortest path from an origin to all other destinations.
    The labels and preds are stored in node instances.
    """

    for n in network.nodeSet:
        network.nodeSet[n].label = np.inf
        network.nodeSet[n].pred = None
    network.nodeSet[origin].label = 0.0
    network.nodeSet[origin].pred = None
    SE = [(0, origin)]
    while SE:
        currentNode = heapq.heappop(SE)[1]
        currentLabel = network.nodeSet[currentNode].label
        for toNode in network.nodeSet[currentNode].outLinks:
            link = (currentNode, toNode)
            newNode = toNode
            newPred = currentNode
            existingLabel = network.nodeSet[newNode].label
            newLabel = currentLabel + network.linkSet[link].cost
            if newLabel < existingLabel:
                heapq.heappush(SE, (newLabel, newNode))
                network.nodeSet[newNode].label = newLabel
                network.nodeSet[newNode].pred = newPred


def BPRcostFunction(optimal: bool,
                    fft: float,
                    alpha: float,
                    flow: float,
                    capacity: float,
                    beta: float,
                    ) -> float:
    if capacity < 1e-3:
        return np.finfo(np.float32).max
    if optimal:
        return fft * (1 + (alpha * math.pow((flow * 1.0 / capacity), beta)) * (beta + 1))
    return fft * (1 + alpha * math.pow((flow * 1.0 / capacity), beta))


def constantCostFunction(optimal: bool,
                         fft: float,
                         alpha: float,
                         flow: float,
                         capacity: float,
                         beta: float,
                         length: float,
                         maxSpeed: float
                         ) -> float:
    if optimal:
        return fft + flow
    return fft


def greenshieldsCostFunction(optimal: bool,
                             fft: float,
                             alpha: float,
                             flow: float,
                             capacity: float,
                             beta: float,
                             length: float,
                             maxSpeed: float
                             ) -> float:
    if capacity < 1e-3:
        return np.finfo(np.float32).max
    if optimal:
        return (length * (capacity ** 2)) / (maxSpeed * (capacity - flow) ** 2)
    return length / (maxSpeed * (1 - (flow / capacity)))


def updateTravelTime(network: FlowTransportNetwork, optimal: bool = False, costFunction=BPRcostFunction):
    """
    This method updates the travel time on the links with the current flow
    """
    if do_mp:
        inputs = []
        for l in network.linkSet:
            link = network.linkSet[l]
            link_tuple = optimal, link.fft, link.alpha, link.flow, link.capacity, link.beta
            inputs.append(link_tuple)
        with multiprocessing.Pool() as pool:
            result = pool.starmap(costFunction, inputs)
        for i, l in enumerate(network.linkSet):
            network.linkSet[l].cost = result[i]
    else:
        for l in network.linkSet:
            network.linkSet[l].cost = costFunction(optimal,
                                                   network.linkSet[l].fft,
                                                   network.linkSet[l].alpha,
                                                   network.linkSet[l].flow,
                                                   network.linkSet[l].capacity,
                                                   network.linkSet[l].beta,
                                                   )


def findAlpha(x_bar, network: FlowTransportNetwork, optimal: bool = False, costFunction=BPRcostFunction):
    """
    This uses unconstrained optimization to calculate the optimal step size required
    for Frank-Wolfe Algorithm
    """

    def df(alpha):
        alpha = max(0, min(1, alpha))
        sum_derivative = 0  # this line is the derivative of the objective function.
        for l in network.linkSet:
            tmpFlow = alpha * x_bar[l] + (1 - alpha) * network.linkSet[l].flow
            tmpCost = costFunction(optimal,
                                   network.linkSet[l].fft,
                                   network.linkSet[l].alpha,
                                   tmpFlow,
                                   network.linkSet[l].capacity,
                                   network.linkSet[l].beta,
                                   )
            sum_derivative = sum_derivative + (x_bar[l] - network.linkSet[l].flow) * tmpCost
        return sum_derivative

    sol = fsolve(df, np.array([0.5]))
    return np.clip(sol[0], 0, 1)


def tracePreds(dest, network: FlowTransportNetwork):
    """
    This method traverses predecessor nodes in order to create a shortest path
    """
    prevNode = network.nodeSet[dest].pred
    spLinks = []
    while prevNode is not None:
        spLinks.append((prevNode, dest))
        dest = prevNode
        prevNode = network.nodeSet[dest].pred
    return spLinks


def loadAON(network: FlowTransportNetwork, computeXbar: bool = True):
    """
    This method produces auxiliary flows for all or nothing loading.
    """
    x_bar = {l: 0.0 for l in network.linkSet}
    SPTT = []
    for r in network.originZones:
        result = AONOriginZone(r, 0.0, network, computeXbar, x_bar)
        SPTT.append(result)

    return sum(SPTT), x_bar


def AONOriginZone(r: int, SPTT: float, network: FlowTransportNetwork, computeXbar: bool, x_bar: dict):
    DijkstraHeap(r, network=network)
    for s in network.zoneSet[r].destList:
        dem = network.tripSet[r, s].demand

        if dem <= 0:
            continue

        SPTT = SPTT + network.nodeSet[s].label * dem

        if computeXbar and r != s:
            for spLink in tracePreds(s, network):
                x_bar[spLink] = x_bar[spLink] + dem
    return SPTT



def readDemand(demand_df: pd.DataFrame, network: FlowTransportNetwork) -> None:
    for index, row in demand_df.iterrows():

        init_node = str(int(row["init_node"]))
        term_node = str(int(row["term_node"]))
        demand = row["demand"]

        network.tripSet[init_node, term_node] = Demand(init_node, term_node, demand)
        if init_node not in network.zoneSet:
            network.zoneSet[init_node] = Zone(init_node)
        if term_node not in network.zoneSet:
            network.zoneSet[term_node] = Zone(term_node)
        if term_node not in network.zoneSet[init_node].destList:
            network.zoneSet[init_node].destList.append(term_node)

    print(len(network.tripSet), "OD pairs")
    print(len(network.zoneSet), "OD zones")


def readNetwork(network_df: pd.DataFrame, network: FlowTransportNetwork):
    for index, row in network_df.iterrows():

        init_node = str(int(row["init_node"]))
        term_node = str(int(row["term_node"]))
        capacity = row["capacity"]
        length = row["length"]
        free_flow_time = row["free_flow_time"]
        b = row["b"]
        power = row["power"]
        speed = row["speed"]
        toll = row["toll"]
        link_type = row["link_type"]

        network.linkSet[init_node, term_node] = Link(init_node=init_node,
                                                     term_node=term_node,
                                                     capacity=capacity,
                                                     length=length,
                                                     fft=free_flow_time,
                                                     b=b,
                                                     power=power,
                                                     speed_limit=speed,
                                                     toll=toll,
                                                     linkType=link_type
                                                     )
        if init_node not in network.nodeSet:
            network.nodeSet[init_node] = Node(init_node)
        if term_node not in network.nodeSet:
            network.nodeSet[term_node] = Node(term_node)
        if term_node not in network.nodeSet[init_node].outLinks:
            network.nodeSet[init_node].outLinks.append(term_node)
        if init_node not in network.nodeSet[term_node].inLinks:
            network.nodeSet[term_node].inLinks.append(init_node)

    print(len(network.nodeSet), "nodes")
    print(len(network.linkSet), "links")


def readNodes(node_file, network):
    for i in node_file.index.values:
        network.nodeSet[str(i)].lat = node_file.loc[i, 'X']
        network.nodeSet[str(i)].lon = node_file.loc[i, 'Y']


def get_TSTT(network: FlowTransportNetwork, costFunction=BPRcostFunction, use_max_capacity: bool = True):
    TSTT = round(sum([network.linkSet[a].flow * costFunction(optimal=False,
                                                             fft=network.linkSet[
                                                                 a].fft,
                                                             alpha=network.linkSet[
                                                                 a].alpha,
                                                             flow=network.linkSet[
                                                                 a].flow,
                                                             capacity=network.linkSet[
                                                                 a].max_capacity if use_max_capacity else network.linkSet[
                                                                 a].capacity,
                                                             beta=network.linkSet[
                                                                 a].beta,
                                                             ) for a in
                      network.linkSet]), 9)
    return TSTT


def assignment_loop(network: FlowTransportNetwork,
                    algorithm: str = "FW",
                    systemOptimal: bool = False,
                    costFunction=BPRcostFunction,
                    accuracy: float = 0.001,
                    maxIter: int = 1000,
                    maxTime: int = 3600,
                    verbose: bool = True,
                    pre_loaded_network: bool = False,
                    multiprocessing: bool = False):
    """
    For explanation of the algorithm see Chapter 7 of:
    https://sboyles.github.io/blubook.html
    PDF:
    https://sboyles.github.io/teaching/ce392c/book.pdf
    """
    connected = check_connectivity(network)
    if not connected:
        network.feasible = False
        network.cost = np.inf
        return network
    else:
        network.feasible = True

    iteration_number = 1
    gap = np.inf
    assignmentStartTime = time.time()
    increasing_alpha = False
    TSTT: float = 1

    if not pre_loaded_network:
        network.reset_flow()

    # Check if desired accuracy is reached
    while gap > accuracy:

        # Get x_bar through all-or-nothing assignment
        _, x_bar = loadAON(network=network)

        if algorithm == "MSA" or iteration_number == 1: # (iteration_number == 1 and pre_loaded_network == False):
            alpha = (1 / iteration_number)
        elif algorithm == "FW":
            # If using Frank-Wolfe determine the step size alpha by solving a nonlinear equation
            alpha = findAlpha(x_bar,
                              network=network,
                              optimal=systemOptimal,
                              costFunction=costFunction)
        else:
            print("Terminating the program.....")
            print("The solution algorithm ", algorithm, " does not exist!")
            raise TypeError('Algorithm must be MSA or FW')
        if alpha <= 1e-6:
            alpha = (0.5 / iteration_number)
            print(f'The FW algorithm set alpha to zero, correcting to {alpha}')

        # Apply flow improvement (not time intensive enough to warrant multiprocessing)
        for l in network.linkSet:
            network.linkSet[l].flow = update_link_flow(alpha, x_bar[l], network.linkSet[l].flow)

        # Compute the new travel time
        updateTravelTime(network=network,
                         optimal=systemOptimal,
                         costFunction=costFunction)

        # Compute the relative gap
        # TODO: We're calling loadAON twice in the same while loop, you can store these results and have to call it less
        old_TSTT = TSTT
        SPTT, _ = loadAON(network=network, computeXbar=False)
        SPTT = round(SPTT, 9)
        TSTT = round(sum([network.linkSet[a].flow * network.linkSet[a].cost for a in
                          network.linkSet]), 9)
        network.cost = TSTT

        cost_change = (TSTT - old_TSTT) / old_TSTT
        gap = (TSTT / SPTT) - 1
        if gap < 0:
            print(f"Error, gap is {gap}. It should never be less than 0.")
            gap = 99
        else:
            print(f'alpha = {round(alpha, 5)}, gap = {round(gap, 5)}, cost change = {round(cost_change*100, 5)}%')

        gap = max(abs(cost_change), gap)
        iteration_number += 1
        if iteration_number > maxIter:
            if verbose:
                print(
                    "The assignment did not converge to the desired gap and the max number of iterations has been reached")
                print("Assignment took", round(time.time() - assignmentStartTime, 5), "seconds")
                print("Current gap:", round(gap, 5))
            return network
        if time.time() - assignmentStartTime > maxTime:
            if verbose:
                print("The assignment did not converge to the desired gap and the max time limit has been reached")
                print("Assignment did ", iteration_number, "iterations")
                print("Current gap:", round(gap, 5))
            return network

    if verbose:
        print("Assignment converged in ", iteration_number, "iterations")
        print("Assignment took", round(time.time() - assignmentStartTime, 5), "seconds")
        print("Current gap:", round(gap, 5))
    return network

def update_link_flow(alpha, x_bar_l, link_l_flow):
    answer = alpha * x_bar_l + (1 - alpha) * link_l_flow
    return answer


def writeResults(network: FlowTransportNetwork, output_file: str, costFunction=BPRcostFunction,
                 systemOptimal: bool = False, verbose: bool = True):
    outFile = open(output_file, "w")
    TSTT = get_TSTT(network=network, costFunction=costFunction)
    if verbose:
        print("\nTotal system travel time:", f'{TSTT} secs')
    tmpOut = "Total Travel Time:\t" + str(TSTT)
    outFile.write(tmpOut + "\n")
    tmpOut = "Cost function used:\t" + BPRcostFunction.__name__
    outFile.write(tmpOut + "\n")
    tmpOut = ["User equilibrium (UE) or system optimal (SO):\t"] + ["SO" if systemOptimal else "UE"]
    outFile.write("".join(tmpOut) + "\n\n")
    tmpOut = "init_node\tterm_node\tflow\ttravelTime"
    outFile.write(tmpOut + "\n")
    for i in network.linkSet:
        tmpOut = str(network.linkSet[i].init_node) + "\t" + str(
            network.linkSet[i].term_node) + "\t" + str(
            network.linkSet[i].flow) + "\t" + str(costFunction(False,
                                                               network.linkSet[i].fft,
                                                               network.linkSet[i].alpha,
                                                               network.linkSet[i].flow,
                                                               network.linkSet[i].max_capacity,
                                                               network.linkSet[i].beta,
                                                               ))
        outFile.write(tmpOut + "\n")
    outFile.close()


def load_network(net_file: str,
                 demand_file: str = None,
                 node_file: str = None,
                 force_net_reprocess: bool = False,
                 verbose: bool = True
                 ) -> FlowTransportNetwork:
    readStart = time.time()
    if isinstance(net_file, pd.DataFrame) and isinstance(demand_file, pd.DataFrame):
        demand_df = demand_file
        net_df = net_file
    else:
        if demand_file is None:
            demand_file = '_'.join(net_file.split("_")[:-1] + ["trips.tntp"])

        net_name = net_file.split("/")[-1].split("_")[0]

        if verbose:
            print(f"Loading network {net_name}...")

        net_df, demand_df = import_network(
            net_file,
            demand_file,
            force_reprocess=force_net_reprocess
        )

    network = FlowTransportNetwork()

    readDemand(demand_df, network=network)
    readNetwork(net_df, network=network)
    if node_file is not None:
        readNodes(node_file, network=network)

    network.originZones = set([k[0] for k in network.tripSet])

    if verbose:
        print("Network loaded")
        print("Reading the network data took", round(time.time() - readStart, 2), "secs\n")

    return network


def computeAssingment(net_file: str,
                      demand_file: str = None,
                      algorithm: str = "FW",  # FW or MSA
                      costFunction=BPRcostFunction,
                      systemOptimal: bool = False,
                      accuracy: float = 0.0001,
                      maxIter: int = 1000,
                      maxTime: int = 60,
                      results_file: str = None,
                      force_net_reprocess: bool = False,
                      verbose: bool = True
                      ) -> float:
    """
    This is the main function to compute the user equilibrium UE (default) or system optimal (SO) traffic assignment
    All the networks present on https://github.com/bstabler/TransportationNetworks following the tntp format can be loaded


    :param net_file: Name of the network (net) file following the tntp format (see https://github.com/bstabler/TransportationNetworks)
    :param demand_file: Name of the demand (trips) file following the tntp format (see https://github.com/bstabler/TransportationNetworks), leave None to use dafault demand file
    :param algorithm:
           - "FW": Frank-Wolfe algorithm (see https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm)
           - "MSA": Method of successive averages
           For more information on how the algorithms work see https://sboyles.github.io/teaching/ce392c/book.pdf
    :param costFunction: Which cost function to use to compute travel time on edges, currently available functions are:
           - BPRcostFunction (see https://rdrr.io/rforge/travelr/man/bpr.function.html)
           - greenshieldsCostFunction (see Greenshields, B. D., et al. "A study of traffic capacity." Highway research board proceedings. Vol. 1935. National Research Council (USA), Highway Research Board, 1935.)
           - constantCostFunction
    :param systemOptimal: Wheather to compute the system optimal flows instead of the user equilibrium
    :param accuracy: Desired assignment precision gap
    :param maxIter: Maximum nuber of algorithm iterations
    :param maxTime: Maximum seconds allowed for the assignment
    :param results_file: Name of the desired file to write the results,
           by default the result file is saved with the same name as the input network with the suffix "_flow.tntp" in the same folder
    :param force_net_reprocess: True if the network files should be reprocessed from the tntp sources
    :param verbose: print useful info in standard output
    :return: Totoal system travel time
    """

    network = load_network(net_file=net_file, demand_file=demand_file, verbose=verbose, force_net_reprocess=force_net_reprocess)

    if verbose:
        print("Computing assignment...")
    network = assignment_loop(network=network, algorithm=algorithm, systemOptimal=systemOptimal, costFunction=costFunction,
                           accuracy=accuracy, maxIter=maxIter, maxTime=maxTime, verbose=verbose)

    if results_file is None:
        results_file = '_'.join(net_file.split("_")[:-1] + ["flow.tntp"])

    writeResults(network=network,
                 output_file=results_file,
                 costFunction=costFunction,
                 systemOptimal=systemOptimal,
                 verbose=verbose)

    return network.cost


if __name__ == '__main__':

    # This is an example usage for calculating System Optimal and User Equilibrium with Frank-Wolfe

    net_file = str(PathUtils.sioux_falls_net_file)

    total_system_travel_time_optimal = computeAssingment(net_file=net_file,
                                                         algorithm="FW",
                                                         costFunction=BPRcostFunction,
                                                         systemOptimal=True,
                                                         verbose=True,
                                                         accuracy=0.00001,
                                                         maxIter=1000,
                                                         maxTime=6000000)

    total_system_travel_time_equilibrium = computeAssingment(net_file=net_file,
                                                             algorithm="FW",
                                                             costFunction=BPRcostFunction,
                                                             systemOptimal=False,
                                                             verbose=True,
                                                             accuracy=0.001,
                                                             maxIter=1000,
                                                             maxTime=6000000)

    print("UE - SO = ", total_system_travel_time_equilibrium - total_system_travel_time_optimal)


def check_connectivity(network) -> bool:
    network_graph = network.to_networkx()
    strongly_connected = nx.is_strongly_connected(network_graph)
    return strongly_connected
