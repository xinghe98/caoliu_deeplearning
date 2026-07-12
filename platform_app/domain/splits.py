import hashlib


def stable_split(content_group_id: str, dataset_role: str | None = None) -> str:
    """Assign a stable split for a content group.

    external_test is locked and never reassigned.
    Other groups hash into 80% train / 10% validation / 10% production_shadow_test.
    """
    if dataset_role == 'external_test':
        return 'external_test'
    bucket = int(hashlib.sha256(content_group_id.encode('utf-8')).hexdigest(), 16) % 10
    if bucket == 0:
        return 'production_shadow_test'
    if bucket == 1:
        return 'validation'
    return 'train'


TRAINABLE_SPLITS = frozenset({'train', 'validation', 'production_shadow_test'})
GRADIENT_SPLITS = frozenset({'train'})
