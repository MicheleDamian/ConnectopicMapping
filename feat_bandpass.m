function out = feat_bandpass(source, tr, lp_sigma, hp_sigma, demean)

    %hp_sigma = hp_sigma * tr;
    hp_max = hp_sigma * 3;
    t_exp = -hp_max : 1/tr : hp_max;
    hp_exp = exp(-(t_exp.^2) / (2 * hp_sigma^2));
    
    hp_half_width = floor(numel(t_exp) / 2);
    
    done_c0 = 0;
    c0 = 0;
    
    %figure(10);
    %plot(t_exp, hp_exp); grid on;
    %title('exp');

    for t = 0:length(source)-1
                 
        A = 0;
        B = 0;
        C = 0;
        D = 0;
        N = 0;
        
        a = [];
        b = [];
        cdt = [];
        d = [];
        n = [];
                    
        for tt = max([t-hp_half_width, 0]):min([t+hp_half_width, length(source)-1])
           
            dt = tt-t;
            dt_i = dt + hp_half_width + 1;
            w = hp_exp(dt_i);
            A = A + w * t_exp(dt_i);
            B = B + w * source(tt+1);
            C = C + w * t_exp(dt_i) * t_exp(dt_i);
            D = D + w * t_exp(dt_i) * source(tt+1);
            N = N + w;
            
            a(dt_i) = w * t_exp(dt_i);
            b(dt_i) = w * source(tt+1);
            cdt(dt_i) = w * t_exp(dt_i) * t_exp(dt_i);
            d(dt_i) = w * t_exp(dt_i) * source(tt+1);
            n(dt_i) = w;
        
        end
        
        tmpdenom = C.*N - A.*A;
                    
        if tmpdenom ~= 0
            c = (B.*C - A.*D) / tmpdenom;
            if ~done_c0
                c0 = c;
	            done_c0 = 1;
            end
	        array2(t+1) = c0 + source(t+1) - c;
            if t == 320
                numel(a)
                source(t-hp_half_width:t+hp_half_width)
                a, b, cdt, d, n
                A, B, C, D, N
                tmpdenom
                B.*C - A.*D
                c0, c
                source(t+1), array2(t+1)
            end
        else
            array2(t+1) = source(t+1);
        end
        
        if t == -100
            figure(11);
            
            x = (0:numel(n)-1)-hp_half_width;
            
            plot(x, n, 'r-+', ...
                 x, a, 'g-o', ... 
            	 x, b, 'y-o', ... 
            	 x, cdt, 'k-o', ...
            	 x, d, 'm-o', ...
            	 x, cdt.*n, 'g-*', ...
            	 x, a.*a, 'y-*', ...
                 x, cdt.*n - a.*a, 'm-*');
            legend('w = hp exp', ...
                'a = w*dt', ...
                'b = w*source(t)', ... 
                'c = w*dt^2', ...
                'd = w*dt*source(t)', ...
                'c*n', ...
                'a*a', ...
                'c*n - a*a');
            grid on;
        end
        
    end
    
    mean_val = mean(array2);
    
    if demean == 1
        out = array2 - mean_val;
    else
        out = array2;
    end
end

