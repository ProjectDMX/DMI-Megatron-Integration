"""User numerical functions receive rows and return row collections; no storage IO."""


def gradient_health(rows):
    # Duplicate transport deliveries retain the same execution identity.
    values = {}
    for row in rows:
        key = (row.model_id, row.global_batch_id, row.attempt_id)
        if key in values and values[key] != row.value:
            raise ValueError('Conflicting gradient norms for one iteration')
        values[key] = row.value
    return ([dict(model_id=run, global_batch_id=batch, attempt_id=attempt,
                  grad_norm=value)
             for (run, batch, attempt), value in values.items()],)
