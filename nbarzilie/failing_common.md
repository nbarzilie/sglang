# Failing Common Status Test

## 1. Failing Test Code

```python
def test_failed_status_overrides_later_non_failed_status(self):
    mgr = object.__new__(CommonKVManager)
    mgr.request_status = {9: KVPoll.Failed}

    CommonKVManager.update_status(mgr, 9, KVPoll.Success)

    self.assertEqual(mgr.request_status[9], KVPoll.Failed)
```

## 2. Failing Test Output

```text
FAILED test/registered/unit/disaggregation/test_nixl_node_failure.py::TestNixlNodeFailure::test_failed_status_overrides_later_non_failed_status

self = <test_nixl_node_failure.TestNixlNodeFailure testMethod=test_failed_status_overrides_later_non_failed_status>

    def test_failed_status_overrides_later_non_failed_status(self):
        mgr = object.__new__(CommonKVManager)
        mgr.request_status = {9: KVPoll.Failed}

        CommonKVManager.update_status(mgr, 9, KVPoll.Success)

>       self.assertEqual(mgr.request_status[9], KVPoll.Failed)
E       AssertionError: 4 != 0

test/registered/unit/disaggregation/test_nixl_node_failure.py:75: AssertionError

During handling of the above exception, another exception occurred:

python/sglang/srt/utils/common.py:2735: Exception: retry() exceed maximum number of retries.

Short test summary:
FAILED test/registered/unit/disaggregation/test_nixl_node_failure.py::TestNixlNodeFailure::test_failed_status_overrides_later_non_failed_status - Exception: retry() exceed maximum number of retries.
1 failed, 79 passed, 4 warnings in 31.65s
```

## 3. Reason For Failure

`CommonKVManager.update_status()` treats request states as monotonic enum values and uses `max(existing_status, new_status)` for non-failed updates.

`KVPoll.Failed` has value `0`, while `KVPoll.Success` has value `4`. If a room is already marked failed and a later success update arrives, `max(KVPoll.Failed, KVPoll.Success)` overwrites the failed state with success.

That makes failure non-terminal in `request_status`.

## 4. Suggested Fix

Make `KVPoll.Failed` terminal in `CommonKVManager.update_status()`: if the existing state is already failed, keep it failed regardless of later non-failed updates.

```python
def update_status(self, bootstrap_room: int, status: KVPoll):
    if bootstrap_room not in self.request_status:
        if status == KVPoll.Failed:
            return
        self.request_status[bootstrap_room] = status
    else:
        if (
            status == KVPoll.Failed
            or self.request_status[bootstrap_room] == KVPoll.Failed
        ):
            self.request_status[bootstrap_room] = KVPoll.Failed
        else:
            self.request_status[bootstrap_room] = max(
                self.request_status[bootstrap_room], status
            )
```
