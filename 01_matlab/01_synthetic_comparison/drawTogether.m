function drawTogether(X, Y, my_xlim, my_ylim, my_zlim, my_view, ...
    Z_orig, Z_estimation, Title, Label)
    % create the first surface using colors from the winter colormap
    fig = figure(); 
    tcl = tiledlayout(fig,1,1); 
    ax = nexttile(tcl); 
    axEMOFS = ax;
    surf1Colormap = winter(); 
    tc = colormapToTruecolor(surf1Colormap,Z_orig);  
    hsurf1 = surf(ax,X,Y,Z_orig,tc,'FaceColor','interp','EdgeAlpha',0.3);
    set(hsurf1, 'FaceAlpha', 0.2);
    % create second surface using colors from the autumn colormap
    hold on
    surf2Colormap = spring(); 
    tc = colormapToTruecolor(surf2Colormap,Z_estimation);
    hsurf2 = surf(ax,X,Y,Z_estimation,tc,'FaceColor','interp','EdgeAlpha',0.3);
    axHidden = axes(tcl,'visible','off','HandleVisibility','off');
    colormap(ax,surf1Colormap)
    colormap(axHidden,surf2Colormap)
    cb1 = colorbar(ax);
    cb1.Layout.Tile = 'east';
    cb1.Label.String = 'Original PDF';
    cb2 = colorbar(axHidden);
    cb2.Layout.Tile = 'east';
    cb2.Label.String = Label;
    xlim([ -my_xlim my_xlim ]);
    ylim([ -my_ylim my_ylim ]);
    zlim([0 my_zlim]);
    view( my_view );
    xlabel('x (in)');
    ylabel('y (in)');
    zlabel(Title);
end