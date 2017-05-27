from ij.plugin.frame import RoiManager
from ij.process import ColorProcessor, ImageConverter, ImageProcessor
from ij.io import TiffDecoder
from ij import IJ, ImagePlus, ImageStack
from ij.plugin.filter import ParticleAnalyzer as PA, ThresholdToSelection
from ij.measure import ResultsTable

#from java.awt.image import BufferedImage

def roi_overlaps(img, rm, i, j):
	rm.setSelectedIndexes([i,j])
	rm.runCommand("AND")
	inter_roi = img.getRoi()
	if i!=j and type(inter_roi) != 'NoneType':
		return True
	else:
		return False

def analyze_particles(imp):
	#imp = IJ.getImage()
	MAXSIZE = 10000
	MINSIZE = 100
	options = PA.SHOW_ROI_MASKS \
		+ PA.EXCLUDE_EDGE_PARTICLES \
		+ PA.INCLUDE_HOLES \
		+ PA.SHOW_RESULTS 
	rt = ResultsTable()
	p = PA(options, PA.AREA + PA.STACK_POSITION, rt, MINSIZE, MAXSIZE)
	p.setHideOutputImage(True)
	stk = ImageStack(imp.getWidth(), imp.getHeight())
	
	for i in range(imp.getStackSize()):
		imp.setSliceWithoutUpdate(i + 1)
		p.analyze(imp)
		mmap = p.getOutputImage()
		stk.addSlice(mmap.getProcessor())
	ImagePlus("tt", stk).show()
	
image_path = '/home/brad/data/ish/170116_deaf_crhbp/' 

# Read in user-defined ROIs
rm = RoiManager.getInstance()
rm.reset()
us_rois = image_path + 'RoiSet.zip'
rm.runCommand('Open', us_rois)
print rm.getCount()

# Get image
image_fname = 'bk27bk53_crhbp_sec2_L_ra.tif'
img = ImagePlus(image_path+image_fname)
#img2 = img.duplicate()
#img2.show()
#-- filter out red pixels--
img_ic = ImageConverter(img)
img_ic.convertToRGB()
img_ic.convertToHSB()
#threshes = [[200,255],[0,255], [0, 200]]
stack = img.getStack()
masks = []
for i in xrange(len(threshes)):
	ip = stack.getProcessor(i+1)
	ip.setAutoThreshold('Default', False, ImageProcessor.NO_LUT_UPDATE)
	#ip.setThreshold(threshes[i][0], threshes[i][1], ImageProcessor.NO_LUT_UPDATE)
	#ip.setBinaryThreshold()
	roi = ThresholdToSelection().convert(ip)
	stack.setRoi(roi)
	masks = masks +  [ImagePlus("Mask", stack.getMask())]
	#maskimp.show()
	print ip
	#print(ip.getMin())
	#stack.setProcessor(ip, i+1)
	#print stack.getProcessor(1).getMin()
	#stack.update(ip)

## TODO get min and max values to persist!
print stack.getProcessor(1).getMin()
#img_ic = ImageConverter(img)
img_ic.convertHSBToRGB()
img.show()

# Analyze particles
#analyze_particles(img)

# Measure
# Intersect ROIs
#nrois = rm.getCount()
#res = []
#for i in xrange(nrois):
	#for j in xrange(nrois):
	#	print i,j
	#	res = res + [roi_overlaps(img, rm, i, j)]
#print res

# WRITE OUT