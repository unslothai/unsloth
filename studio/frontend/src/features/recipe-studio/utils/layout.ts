import dagre from "@dagrejs/dagre";
import type { Edge, Node } from "@xyflow/react";
import type { LayoutDirection } from "../types";

type LayoutOptions = {
  direction?: LayoutDirection;
  nodesep?: number;
  ranksep?: number;
  edgesep?: number;
  nodeWidth?: number;
  nodeHeight?: number;
};

export function getLayoutedElements<TNode extends Node>(
  nodes: TNode[],
  edges: Edge[],
  options: LayoutOptions = {},
): { nodes: TNode[]; edges: Edge[] } {
  const {
    direction = "LR",
    nodesep = 80,
    ranksep = 80,
    edgesep = 28,
    nodeWidth = 220,
    nodeHeight = 64,
  } = options;

  const graph = new dagre.graphlib.Graph();
  graph.setDefaultEdgeLabel(() => ({}));
  graph.setGraph({
    rankdir: direction,
    nodesep,
    ranksep,
    edgesep,
    ranker: "network-simplex",
  });

  nodes.forEach((node) => {
    const width = node.measured?.width ?? nodeWidth;
    const height = node.measured?.height ?? nodeHeight;
    graph.setNode(node.id, { width, height });
  });

  edges.forEach((edge) => {
    const semantic = edge.type === "semantic";
    const aux = edge.source.startsWith("aux-") || edge.target.startsWith("aux-");
    graph.setEdge(edge.source, edge.target, {
      minlen: semantic ? 1 : 1,
      weight: semantic ? 10 : aux ? 1 : 3,
    });
  });

  dagre.layout(graph);

  const layoutedNodes = nodes.map((node) => {
    const pos = graph.node(node.id);
    const width = node.measured?.width ?? nodeWidth;
    const height = node.measured?.height ?? nodeHeight;
    return {
      ...node,
      position: {
        x: pos.x - width / 2,
        y: pos.y - height / 2,
      },
    };
  });

  return { nodes: layoutedNodes, edges };
}
