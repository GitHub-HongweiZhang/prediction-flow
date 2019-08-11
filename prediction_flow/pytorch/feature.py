"""
DeepFM.
"""


class EmbeddingRef(object):
    """ This class is used to identify which features share
    embedding weights.

    Parameters
    ----------
    refs : dict
        Example :
            {'clicked_item_ids': 'item_id',
             'clicked_item_categories': 'item_category'}
    """
    def __init__(self, refs=None):
        self.refs = refs

    def share_with_others(self, feature_name):
        if self.refs and feature_name in self.refs:
            return True

        return False

    def ref_feature_name(self, feature_name):
        return self.refs[feature_name]
