function hout = subimagesc(varargin)
%SUBIMAGE Display multiple images in single figure.
%   You can use SUBIMAGESC in conjunction with SUBPLOT to create
%   figures with multiple images, even if the images have
%   different colormaps. SUBIMAGESC works by converting images to
%   truecolor for display purposes, thus avoiding colormap
%   conflicts.
%
%   SUBIMAGE(X,MAP) displays the indexed image X with colormap
%   MAP in the current axes.
%
%   SUBIMAGE(X,MAP,CLIM) displays the indexed image X with colormap
%   MAP in the current axes, with values clipped with CLIM
%
%   SUBIMAGE(x,y,...) displays an image with nondefault spatial
%   coordinates.
%
%   H = SUBIMAGE(...) returns a handle to the image object.
%
%   Class Support
%   -------------
%   The input image can be of class logical, uint8, uint16,
%   or double.
%
%   Example
%   -------
%       load trees
%       [X2,map2] = imread('forest.tif');
%       subplot(1,2,1), subimage(X,map)
%       subplot(1,2,2), subimage(X2,map2)
%
%   See also IMSHOW, SUBPLOT.

%   Copyright 1993-2011 The MathWorks, Inc.

[x,y,cdata] = parse_inputs(varargin{:});

ax = newplot;
fig = ancestor(ax,'figure');
cm = get(fig,'Colormap');

% Go change all the existing image and texture-mapped surface 
% objects to truecolor.
% h = [findobj(fig,'Type','image') ; 
%     findobj(fig,'Type','surface','FaceColor','texturemap')];
% for k = 1:length(h)
%     if (ndims(get(h(k), 'CData')) < 3)
% 
%         X = get(h(k),'CData');
%         
%         if strcmp(get(h(k), 'CDataMapping'), 'scaled')
%             clim = get(ancestor(h(k),'axes'),'CLim');
%             X = scaledind2ind(X,cm,clim);
%         end
% 
%         if strcmp(get(h(k),'Type'),'image')
%             set(h(k), 'CData', ...
%                 iptgate('ind2rgb8',X,cm));
%         else
%             set(h(k), 'CData', ind2rgb(X, cm));
%         end
% 
%     end
% end

h = image(x, y, cdata);
%axis image;

if nargout==1,
    hout = h;
end

%--------------------------------------------------------
% Subfunction PARSE_INPUTS
%--------------------------------------------------------
function [x,y,cdata] = parse_inputs(varargin)

x = [];
y = [];

scaled = 0;
binary = 0;

switch nargin
    case 0
        error(message('images:subimage:notEnoughInputs'))
        
    case 1
        error(message('images:subimage:notEnoughInputs'))
        
    case 2
        % subimagesc(X,map)
        cdata = varargin{1};
        if (size(varargin{2},2) == 3)
            cmap = varargin{2};
        else
            error(message('images:subimage:invalidInputs'))
        end
        
        clim = [min(varargin{1}(:)), max(varargin{1}(:))];
        scaled = 1;
        
    case 3
        % subimage(X,map,clim)
        cdata = varargin{1};
        if (isequal(size(varargin{3}), [1 2]))
            clim = varargin{3};
            if (clim(1) == clim(2))
                error(message('images:subimage:aEqualsB'))
            end
            scaled = 1;
        else
            error(message('images:subimage:invalidInputs'))
        end
        
        if (size(varargin{2},2) == 3)
            cmap = varargin{2};
        else
            error(message('images:subimage:invalidInputs'))
        end
        
    case 4
        % subimage(x,y,X,map)
        x = varargin{1};
        y = varargin{2};
        cdata = varargin{3};
        
        if (size(varargin{4},2) == 3)
            cmap = varargin{4};
        else
            error(message('images:subimage:invalidInputs'))
        end
        
        clim = [min(varargin{1}(:)), max(varargin{1}(:))];
        scaled = 1;
        
    case 5
        % subimage(x,y,X,map,clim)
        x = varargin{1};
        y = varargin{2};
        cdata = varargin{3};
        if (isequal(size(varargin{5}), [1 2]))
            clim = varargin{5};
            if (clim(1) == clim(2))
                error(message('images:subimage:aEqualsB'))
            end
            scaled = 1;
        else
            error(message('images:subimage:invalidInputs'))
        end
        
        if (size(varargin{4},2) == 3)
            cmap = varargin{4};
        else
            error(message('images:subimage:invalidInputs'))
        end
        
    otherwise
        error(message('images:subimage:tooManyInputs'))
        
end

if (scaled)
    if (isa(cdata,'double'))
        cdata = (cdata - clim(1)) / (clim(2) - clim(1));
        cdata = min(max(cdata,0),1);
        cdata = round(cdata.*size(cmap,1))+1;
        cdata = ind2rgb(cdata,cmap);
    elseif (isa(cdata,'uint8'))
        cdata = (cdata - clim(1)) / (clim(2) - clim(1));
        cdata = ind2rgb(cdata,cmap);
    else
        error(message('images:subimage:invalidClass'))
        
    end
end

if (isempty(x))
    x = [1 size(cdata,2)];
    y = [1 size(cdata,1)];
end

% Regardless of the input type, at this point in the code,
% cdata represents an RGB image; atomatically clip double RGB images 
% to [0 1] range
if isa(cdata, 'double')
   cdata(cdata > 1) = 1;
   cdata(cdata < 0) = 0;
end
