# Areas for Improvement

- **Automated Testing:** Adding a dedicated testing framework with unit and integration tests would significantly improve the project's reliability.

- **Configuration Flexibility:** The reliance on `config.py` could be reduced by allowing users to pass arguments through the command line for greater flexibility.

- **Code Duplication:** In the `trackers` folder, the `iterable.py` file is duplicated. This could be resolved by creating a shared module to reduce redundancy.

- **Dependency Management:** The `requirements.txt` file is quite large. It could be streamlined by removing unused libraries and organizing the dependencies more efficiently.
