## planned workflow:
1. think about all the tests required in order to stablizie and assure nixl backend with sglang.
2. create nbarzilie/plan.md
2. write down to plan.md in guidelines the testing concepts and cases and group them (unit, basic DP, or any other grouping you find relevant). you can list suggestion of test files and their natural path. 
3. write down to plan.md the flow plan: first testing locally as single script, then as part of ci, locally, then in GitHub actions.
4. write to plan.md a guide for running the tests locally and add setup instructions for each (required gpus etc.)
5. write down to plan.md any other relvant information you find before we move on to writing down the test one by one.

## suites:
### planned suite (for functional and basic tests): base-b-test-2-gpu-large	2-gpu-h100 
### planned suite (for large scale disaggregation and radix cache tests): base-c-test-8-gpu-h20 8-gpu-h20
