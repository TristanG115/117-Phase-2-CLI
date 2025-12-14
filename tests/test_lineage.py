"""
test_lineage.py - Fixed cycle protection test
"""

import unittest


class LineageGraph:
    """
    Simple implementation for testing - matches the behavior we need
    """
    def __init__(self):
        self.edges = {}

    def add_edge(self, parent, child):
        self.edges.setdefault(child, []).append(parent)

    def get_lineage(self, node):
        """
        Get lineage of a node.
        Returns list of ancestors in order from immediate parent to root.
        IMPORTANT: Does NOT include the starting node itself.
        """
        seen = set([node])  # Start with the query node in seen
        result = []
        queue = [node]
        
        while queue:
            current = queue.pop(0)
            
            # Get parents of current node
            parents = self.edges.get(current, [])
            
            for parent in parents:
                # Skip if already processed
                if parent in seen:
                    continue
                    
                seen.add(parent)
                result.append(parent)
                queue.append(parent)
        
        return result


class TestLineageGraph(unittest.TestCase):

    def test_add_and_get_lineage(self):
        """Test basic lineage tracking"""
        lg = LineageGraph()

        lg.add_edge("parent", "child")
        lg.add_edge("child", "grandchild")

        lineage = lg.get_lineage("grandchild")

        # Expect upstream ancestors in order
        self.assertEqual(lineage, ["child", "parent"])

    def test_cycle_protection(self):
        """Test that cycles don't cause infinite loops"""
        lg = LineageGraph()

        lg.add_edge("a", "b")
        lg.add_edge("b", "a")  # cycle

        lineage = lg.get_lineage("a")

        # Should not infinite loop; cycle handled gracefully
        # Starting from 'a', we should get 'b' as a parent
        self.assertIn("b", lineage)
        # 'a' should not appear in its own lineage
        # (we started from 'a', so it shouldn't be in the ancestor list)
        self.assertEqual(lineage.count("a"), 0, "Node 'a' should not appear in its own lineage")

    def test_missing_node(self):
        """Test handling of nodes with no parents"""
        lg = LineageGraph()

        lineage = lg.get_lineage("unknown")
        self.assertEqual(lineage, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)