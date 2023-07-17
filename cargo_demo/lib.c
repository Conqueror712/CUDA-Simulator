CUresult redirect_cuInit(unsigned int flags) {
    mock_cuInit(flags);
    return actual_cuInit(flags);
}