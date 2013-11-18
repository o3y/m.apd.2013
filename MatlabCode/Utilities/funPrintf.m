function silent_fprintf(flag_silent, varargin)
    if ~flag_silent
        fprintf(varargin{:});
    end
end