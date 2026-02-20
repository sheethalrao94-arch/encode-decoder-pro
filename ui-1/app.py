from flask import Flask, render_template, request, jsonify
import heapq
import math
from collections import Counter
import matplotlib
matplotlib.use('Agg')   # IMPORTANT for Flask
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import matplotlib
import networkx as nx



app = Flask(__name__)

# ==========================================================
# ======================== LZW =============================
# ==========================================================

def lzw_encode_with_table(input_string):

    unique_chars = sorted(set(input_string))
    dictionary = {char: i+1 for i, char in enumerate(unique_chars)}

    next_code = len(dictionary) + 1
    table = []
    encoded_output = []
    step = 1

    for char, code in dictionary.items():
        table.append([step, "-", "-", "-", "-", f"{char}->{code}"])
        step += 1

    w = ""

    for c in input_string:
        wc = w + c

        if wc in dictionary:
            table.append([step, w, c, wc, "-", "-"])
            w = wc
        else:
            output_code = dictionary[w]
            encoded_output.append(output_code)

            dictionary[wc] = next_code

            table.append([step, w, c, wc, output_code, f"{wc}->{next_code}"])

            next_code += 1
            w = c

        step += 1

    if w:
        encoded_output.append(dictionary[w])
        table.append([step, w, "-", "-", dictionary[w], "-"])

    stats = {
        "original": len(input_string),
        "encoded": len(encoded_output),
        "ratio": round(len(input_string)/len(encoded_output),3),
        "percentage": round(((len(input_string)-len(encoded_output))/len(input_string))*100,2)
    }

    init_dict = {i+1:ch for i,ch in enumerate(unique_chars)}

    return table, encoded_output, stats, init_dict


def lzw_decode_with_table(encoded_list, dictionary):

    dictionary = dictionary.copy()
    next_code = len(dictionary) + 1
    table = []
    step = 1

    for code, string in dictionary.items():
        table.append([step, "-", "-", "-", "-", code, string])
        step += 1

    first_code = encoded_list[0]
    w = dictionary[first_code]
    output_string = w

    table.append([step, "-", first_code, w, output_string, "-", "-"])
    step += 1

    for k in encoded_list[1:]:

        if k in dictionary:
            entry = dictionary[k]
        else:
            entry = w + w[0]

        output_string += entry

        new_string = w + entry[0]
        dictionary[next_code] = new_string

        table.append([step, w, k, entry, output_string, next_code, new_string])

        next_code += 1
        w = entry
        step += 1

    stats = {
        "original": len(output_string),
        "encoded": len(encoded_list),
        "ratio": round(len(output_string)/len(encoded_list),3),
        "percentage": round(((len(output_string)-len(encoded_list))/len(output_string))*100,2)
    }

    return table, output_string, stats

# ==========================================================
# ================== STATIC HUFFMAN ========================
# ==========================================================
static_tree = None

class HuffmanNode:
    def __init__(self, symbol, prob):
        self.symbol = symbol
        self.prob = prob
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.prob < other.prob


def generate_huffman_codes(node, code="", codes=None):

    if codes is None:
        codes = {}

    if node.symbol is not None:
        codes[node.symbol] = code
        return codes

    generate_huffman_codes(node.left, code + "0", codes)
    generate_huffman_codes(node.right, code + "1", codes)

    return codes


def static_huffman_decode(encoded_string, root):

    decoded_output = ""
    current = root

    for bit in encoded_string:
        current = current.left if bit == "0" else current.right

        if current.symbol is not None:
            decoded_output += current.symbol
            current = root

    return decoded_output


# ================= TREE HELPERS =================

def add_edges(graph, node):
    if node.left:
        graph.add_edge(id(node), id(node.left), label="0")
        add_edges(graph, node.left)

    if node.right:
        graph.add_edge(id(node), id(node.right), label="1")
        add_edges(graph, node.right)


def get_labels(node, labels):
    if node.symbol is not None:
        labels[id(node)] = f"{node.symbol}\n{round(node.prob,3)}"
    else:
        labels[id(node)] = f"{round(node.prob,3)}"

    if node.left:
        get_labels(node.left, labels)
    if node.right:
        get_labels(node.right, labels)


def hierarchy_pos(G, root, width=1.0, vert_gap=0.15, vert_loc=0, xcenter=0.5):
    pos = {root: (xcenter, vert_loc)}
    children = list(G.successors(root))

    if children:
        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2

        for child in children:
            nextx += dx
            pos.update(
                hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx
                )
            )
    return pos



def static_huffman_encode(symbols, frequencies, message):

    heap = []
    merge_steps = []

    # Build heap
    for s, p in zip(symbols, frequencies):
        heapq.heappush(heap, HuffmanNode(s, float(p)))

    step = 1

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        merged_prob = left.prob + right.prob

        merge_steps.append([
            step,
            left.symbol if left.symbol else "Node",
            round(left.prob,4),
            right.symbol if right.symbol else "Node",
            round(right.prob,4),
            round(merged_prob,4)
        ])

        parent = HuffmanNode(None, merged_prob)
        parent.left = left
        parent.right = right

        heapq.heappush(heap, parent)
        step += 1

    root = heap[0]

    # Store tree globally
    global static_tree
    static_tree = root

    # Generate Codes
    codes = generate_huffman_codes(root)

    # Encode
    encoded_msg = "".join(codes[ch] for ch in message)


    # Decode
    decoded_msg = static_huffman_decode(encoded_msg, root)

    # Information Theory
    Lavg = sum(float(p) * len(codes[s]) for s,p in zip(symbols,frequencies))
    H = -sum(float(p) * math.log2(float(p)) for p in frequencies)

    efficiency = H / Lavg
    redundancy = 1 - efficiency

    # ===== TREE GRAPH =====
    G = nx.DiGraph()
    add_edges(G, root)

    labels = {}
    get_labels(root, labels)

    pos = hierarchy_pos(G, id(root))

    plt.figure(figsize=(10,8))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=2000)

    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Static Huffman Tree")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return {
        "merge_steps": merge_steps,
        "codes": codes,
        "encoded": encoded_msg,
        "decoded": decoded_msg,
        "Lavg": round(Lavg,4),
        "Entropy": round(H,4),
        "Efficiency": round(efficiency,4),
        "Redundancy": round(redundancy,4),
        "graph": graph_url
    }

# ==========================================================
# ================== DYNAMIC HUFFMAN ============================
# ==========================================================
class FGKNode:
    def __init__(self, symbol=None, weight=0, order=0):
        self.symbol = symbol
        self.weight = weight
        self.order = order
        self.left = None
        self.right = None
        self.parent = None


class FGK:

    def __init__(self):
        self.max_order = 512
        self.root = FGKNode(order=self.max_order)
        self.NYT = self.root
        self.nodes = {}
        self.next_order = self.max_order - 1


    # ================= BASIC UTILS =================

    def get_code(self, node):
        code = ""
        while node.parent:
            if node.parent.left == node:
                code = "0" + code
            else:
                code = "1" + code
            node = node.parent
        return code


    def is_ancestor(self, a, b):
        while b:
            if b == a:
                return True
            b = b.parent
        return False


    def find_highest(self, weight):
        stack = [self.root]
        candidate = None

        while stack:
            node = stack.pop()
            if node.weight == weight:
                if candidate is None or node.order > candidate.order:
                    candidate = node
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)

        return candidate


    def swap(self, n1, n2):

        if n1 == n2:
            return

        if self.is_ancestor(n1, n2) or self.is_ancestor(n2, n1):
            return

        p1, p2 = n1.parent, n2.parent

        if p1.left == n1:
            p1.left = n2
        else:
            p1.right = n2

        if p2.left == n2:
            p2.left = n1
        else:
            p2.right = n1

        n1.parent, n2.parent = p2, p1
        n1.order, n2.order = n2.order, n1.order


    def update(self, node):
        while node:
            highest = self.find_highest(node.weight)

            if highest and highest != node and highest != node.parent:
                self.swap(node, highest)

            node.weight += 1
            node = node.parent


    def insert(self, symbol):

        internal = FGKNode(weight=0, order=self.next_order)
        self.next_order -= 1

        leaf = FGKNode(symbol=symbol, weight=1, order=self.next_order)
        self.next_order -= 1

        internal.left = self.NYT
        internal.right = leaf
        internal.parent = self.NYT.parent

        if self.NYT.parent:
            if self.NYT.parent.left == self.NYT:
                self.NYT.parent.left = internal
            else:
                self.NYT.parent.right = internal
        else:
            self.root = internal

        self.NYT.parent = internal
        leaf.parent = internal

        self.nodes[symbol] = leaf
        self.update(internal)


    # ================= TREE DRAWING =================

    def add_edges(self, G, node):
        if node.left:
            G.add_edge(id(node), id(node.left))
            self.add_edges(G, node.left)
        if node.right:
            G.add_edge(id(node), id(node.right))
            self.add_edges(G, node.right)


    def get_labels(self, node, labels):
        if node.symbol:
            labels[id(node)] = f"{node.symbol}\nW:{node.weight}\nO:{node.order}"
        else:
            labels[id(node)] = f"W:{node.weight}\nO:{node.order}"

        if node.left:
            self.get_labels(node.left, labels)
        if node.right:
            self.get_labels(node.right, labels)


    def hierarchy_pos(self, G, root, width=1., vert_gap=0.2,
                      vert_loc=0, xcenter=0.5):

        pos = {root: (xcenter, vert_loc)}
        children = list(G.successors(root))

        if children:
            dx = width / len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos.update(self.hierarchy_pos(
                    G, child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc-vert_gap,
                    xcenter=nextx))
        return pos


    def draw_tree(self):

        G = nx.DiGraph()
        self.add_edges(G, self.root)

        labels = {}
        self.get_labels(self.root, labels)

        pos = self.hierarchy_pos(G, id(self.root))

        plt.figure(figsize=(6,4))

        nx.draw(G, pos,
                labels=labels,
                with_labels=True,
                node_color="lightblue",
                node_size=1000,
                font_size=6)

        plt.title("Dynamic Huffman Tree", fontsize=8)

        img = io.BytesIO()
        plt.savefig(img, format='png',bbox_inches='tight', dpi=100)
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return graph_url


    # ================= ENCODE WITH FULL STEPS =================

    def encode(self, text):

        encoded = ""
        steps = []
        step_no = 1

        for ch in text:

            step_info = {}
            step_info["step"] = step_no
            step_info["symbol"] = ch

            if ch in self.nodes:
                node = self.nodes[ch]
                code = self.get_code(node)
                encoded += code
                step_info["type"] = "existing"
                step_info["code"] = code
                self.update(node)

            else:
                nyt_code = self.get_code(self.NYT)
                ascii_code = format(ord(ch), '08b')
                full_code = nyt_code + ascii_code
                encoded += full_code
                step_info["type"] = "new"
                step_info["code"] = full_code
                self.insert(ch)

            step_info["tree"] = self.draw_tree()
            steps.append(step_info)

            step_no += 1

        return encoded, steps



    # ================= DECODE =================
    def decode(self, encoded):

        decoded = ""
        node = self.root
        i = 0

        while i < len(encoded):

            # If leaf (existing symbol)
            if node.symbol is not None:
                decoded += node.symbol
                self.update(node)
                node = self.root
                continue

            # If NYT → read next 8 bits
            if node == self.NYT:

                if i + 8 > len(encoded):
                    break

                symbol = chr(int(encoded[i:i+8], 2))
                i += 8

                decoded += symbol
                self.insert(symbol)
                node = self.root
                continue

            # Traverse
            if encoded[i] == "0":
                node = node.left
            else:
                node = node.right

            i += 1

        return decoded




# ==========================================================
# ================== ARITHMETIC ============================
# ==========================================================

def cumulative_probabilities(probabilities):

    cumulative = {}
    low = 0.0

    for symbol in sorted(probabilities.keys()):
        high = low + probabilities[symbol]
        cumulative[symbol] = (low, high)
        low = high

    return cumulative


def arithmetic_encode(message):

    freq = Counter(message)
    total = sum(freq.values())

    probabilities = {k: v/total for k,v in freq.items()}
    cumulative = cumulative_probabilities(probabilities)

    #  Generate Graph
    graph = generate_arithmetic_graph(message, probabilities)

    low = 0.0
    high = 1.0

    step_table = []

    for level, symbol in enumerate(message):

        range_width = high - low
        sym_low, sym_high = cumulative[symbol]

        new_low = low + range_width * sym_low
        new_high = low + range_width * sym_high

        step_table.append([
            level + 1,
            symbol,
            round(low,6),
            round(high,6),
            round(new_low,6),
            round(new_high,6)
        ])

        low = new_low
        high = new_high

    encoded_value = (low + high) / 2

    # Compression stats
    original_bits = len(message) * 8
    interval_width = high - low
    encoded_bits = math.ceil(-math.log2(interval_width))

    stats = {
        "original_bits": original_bits,
        "encoded_bits": encoded_bits,
        "ratio": round(original_bits/encoded_bits,3),
        "percentage": round((1 - encoded_bits/original_bits)*100,2)
    }

    return {
        "step_table": step_table,
        "encoded_value": round(encoded_value,8),
        "stats": stats,
        "probabilities": probabilities,
        "cumulative": cumulative,
        "length": len(message),
        "graph": graph
    }



def arithmetic_decode(encoded_value, length, cumulative):

    decoded = ""

    for _ in range(length):

        for symbol, (low, high) in cumulative.items():
            if low <= encoded_value < high:
                decoded += symbol
                encoded_value = (encoded_value - low) / (high - low)
                break

    return decoded

def generate_arithmetic_graph(word, probabilities):

    # Use sorted keys (dynamic support)
    symbols = sorted(probabilities.keys())

    cum_ranges = {}
    current_low = 0.0

    for s in symbols:
        p = probabilities[s]
        cum_ranges[s] = (current_low, current_low + p)
        current_low += p

    stages = [{'low': 0.0, 'high': 1.0, 'char': 'Initial'}]
    low, high = 0.0, 1.0

    for char in word:
        r = high - low
        s_low_rel, s_high_rel = cum_ranges[char]
        new_low = low + r * s_low_rel
        new_high = low + r * s_high_rel
        stages.append({'low': new_low, 'high': new_high, 'char': char})
        low, high = new_low, new_high

    fig, ax = plt.subplots(figsize=(14, 8))
    num_stages = len(stages)
    bar_width = 0.3

    for i in range(num_stages):
        curr = stages[i]
        s_low_abs, s_high_abs = curr['low'], curr['high']
        s_range_abs = s_high_abs - s_low_abs

        ax.add_patch(
            plt.Rectangle(
                (i - bar_width/2, 0),
                bar_width,
                1,
                fill=False,
                edgecolor='black',
                lw=2
            )
        )

        for s in symbols:
            s_low_rel, s_high_rel = cum_ranges[s]

            ax.hlines(
                s_high_rel,
                i - bar_width/2,
                i + bar_width/2,
                colors='black',
                lw=1
            )

            ax.text(
                i,
                (s_low_rel + s_high_rel) / 2,
                s,
                ha='center',
                va='center',
                fontsize=12,
                fontweight='bold'
            )

            abs_val = s_low_abs + s_range_abs * s_low_rel

            ax.text(
                i - bar_width/2 - 0.02,
                s_low_rel,
                f"{abs_val:.4f}",
                ha='right',
                va='center',
                fontsize=9
            )

            if s == symbols[-1]:
                ax.text(
                    i - bar_width/2 - 0.02,
                    1.0,
                    f"{s_high_abs:.4f}",
                    ha='right',
                    va='center',
                    fontsize=9
                )

        if i < num_stages - 1:
            next_char = stages[i+1]['char']
            c_low_rel, c_high_rel = cum_ranges[next_char]

            ax.plot(
                [i + bar_width/2, i+1 - bar_width/2],
                [c_low_rel, 0],
                'r--',
                alpha=0.6
            )

            ax.plot(
                [i + bar_width/2, i+1 - bar_width/2],
                [c_high_rel, 1],
                'r--',
                alpha=0.6
            )

    ax.set_xticks(np.arange(num_stages))
    ax.set_xticklabels([s['char'] for s in stages])
    ax.set_yticks([])
    ax.set_title(f"Arithmetic Coding Graph for '{word}'", fontsize=16)

    plt.tight_layout()

    # Convert to Base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return graph_url


# ==========================================================
# ======================== ROUTES ==========================
# ==========================================================

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/lzw')
def lzw_page():
    return render_template("lzw.html")


@app.route('/static_huffman')
def static_huffman():
    return render_template("static_huffman.html")


@app.route('/arithmetic')
def arithmetic():
    return render_template("arithmetic.html")

@app.route('/dynamic_huffman')
def dynamic_huffman():
    return render_template("dynamic_huffman.html")

# ================= LZW ROUTES =================

@app.route('/lzw_encode', methods=['POST'])
def lzw_encode_route():

    text = request.json['input']
    table, output, stats, init_dict = lzw_encode_with_table(text)

    return jsonify({
        "table": table,
        "output": output,
        "stats": stats,
        "init_dict": init_dict
    })


@app.route('/lzw_decode', methods=['POST'])
def lzw_decode_route():

    codes = list(map(int, request.json['input'].split(',')))
    init_dict = {int(k):v for k,v in request.json['init_dict'].items()}

    table, output, stats = lzw_decode_with_table(codes, init_dict)

    return jsonify({
        "table": table,
        "output": output,
        "stats": stats
    })


# ================= statsic HUFFMAN ROUTES =================

huffman_tree_storage = None

@app.route('/static_huffman_encode', methods=['POST'])
def static_huffman_encode_route():

    data = request.json

    symbols = data['symbols'].split(',')
    frequencies = list(map(float, data['frequencies'].split(',')))
    message = data['message']

    result = static_huffman_encode(symbols, frequencies, message)

    return jsonify(result)


@app.route('/static_huffman_decode', methods=['POST'])
def static_decode_route():

    global static_tree

    encoded_string = request.json['input']
    decoded = static_huffman_decode(encoded_string, static_tree)

    return jsonify({"decoded": decoded})

# ================= DYNAMIC HUFFMAN ROUTES =================

@app.route('/dynamic_huffman_encode', methods=['POST'])
def dynamic_huffman_encode():

    text = request.json['input']

    fgk = FGK()
    encoded, steps = fgk.encode(text)

    stats = {
        "original": len(text)*8,
        "encoded": len(encoded),
        "ratio": round((len(text)*8)/len(encoded),3),
        "percentage": round((1 - len(encoded)/(len(text)*8))*100,2)
    }

    return jsonify({
        "output": encoded,
        "stats": stats,
        "steps": steps
    })


@app.route('/dynamic_huffman_decode', methods=['POST'])
def dynamic_huffman_decode():

    encoded = request.json['input']

    fgk = FGK()
    decoded = fgk.decode(encoded)

    tree_image = fgk.draw_tree()

    return jsonify({
        "output": decoded,
        "tree": tree_image
    })



# ================= ARITHMETIC ROUTES =================

@app.route('/arithmetic_encode', methods=['POST'])
def arithmetic_encode_route():

    text = request.json['input']
    result = arithmetic_encode(text)

    return jsonify(result)



@app.route('/arithmetic_decode', methods=['POST'])
def arithmetic_decode_route():

    encoded_value = float(request.json['encoded_value'])
    cumulative = {k: tuple(v) for k,v in request.json['cumulative'].items()}
    length = int(request.json['length'])

    decoded = arithmetic_decode(encoded_value, length, cumulative)

    return jsonify({"output": decoded})


if __name__ == "__main__":
    app.run(debug=True)
