from ij.plugin.frame import RoiManager
from ij.process import ColorProcessor, ImageConverter, ImageProcessor
from ij.io import TiffDecoder
from ij import IJ, ImagePlus, ImageStack
from ij.plugin.filter import ParticleAnalyzer as PA, ThresholdToSelection, Analyzer
from ij.measure import ResultsTable
from ij.gui import ShapeRoi

import re, os

#from java.awt.image import BufferedImage

def roi_overlaps(img, rm, i, j):
	rm.setSelectedIndexes([i,j])
	rm.runCommand("AND")
	inter_roi = img.getRoi()
	if i!=j and (inter_roi is not None):
		return True
	else:
		return False

def analyze_particles(imp, rm, outdir):
	#imp = IJ.getImage()
	MAXSIZE = 10000
	MINSIZE = 0
	options = PA.SHOW_OUTLINES \
		+ PA.EXCLUDE_EDGE_PARTICLES \
		+ PA.INCLUDE_HOLES \
		+ PA.SHOW_RESULTS \
		+ PA.ADD_TO_MANAGER
	measures = PA.MEAN \
		+ PA.AREA \ 
	rt = ResultsTable()
	p = PA(options, measures, rt, MINSIZE, MAXSIZE)
	p.setHideOutputImage(True)
	p.setRoiManager(rm)
	stk = ImageStack(imp.getWidth(), imp.getHeight())


	for i in range(imp.getStackSize()):
		imp.setSliceWithoutUpdate(i + 1)
		p.analyze(imp)
		mmap = p.getOutputImage()
		stk.addSlice(mmap.getProcessor())
	img_part = ImagePlus("tt", stk)
	IJ.saveAs(img_part, "TIFF", outdir + "_analyze_particles.tif")
	return rt

def mask_on_blue(img):
	img_ic = ImageConverter(img)
	img_ic.convertToRGB()
	img_ic.convertToHSB()
	
	threshes = [[60.0,255.0],[0.0,255.0], [0.0, 200.0]]
	stack = img.getStack()
	rois = []
	for i in xrange(len(threshes)):
		ip = stack.getProcessor(i+1)
		ip.setThreshold(threshes[i][0], threshes[i][1], ImageProcessor.NO_LUT_UPDATE)
		#print ip.getMinThreshold(), ip.getMaxThreshold()
		roi = ThresholdToSelection().convert(ip)
		rois = rois + [ShapeRoi(roi)]

	print "Intersecting masking ROIs..."
	rois1 = rois[0]
	for i in range(len(rois))[1:]:
		rois1 = rois1.and(rois[i])
	
	print "Mask original image..."
	#masked = img.duplicate()
	img.setRoi(rois1)
	print img.getDimensions()
	newm = ImagePlus("Mask", img.getProcessor())
	newm.getProcessor().setMask(img.getMask())
	#newm.getProcessor().autoThreshold()
	print newm.getDimensions()
	#newm.setProcessor(img.getMask())
	newm = ImagePlus("Mask", img.getMask())
	newm.changes = False
	newm.getProcessor().invertLut()
	newm.show()
	return newm
	
def analyze_ish_image(image_path, image_fname):

	outdir = "/".join([image_path, "analysis", image_fname])
	
	# Get image
	img = ImagePlus(image_path+image_fname)
	print img.getDimensions()
	rm = RoiManager.getInstance()
	rm.reset()
	
	#-- filter out red pixels--
	newm = mask_on_blue(img)
	print newm.getDimensions()
	img.close()
	# Analyze particles
	rm.reset()
	res = analyze_particles(newm, rm, outdir)
	newm.close()
	# write ROIs for image
	print "Write out detected ROIs..."
	rm.runCommand('Save', image_path + image_fname + ".roi.zip")

	n_sub_rois = rm.getCount()
	print "Number of ROIs: " + str(n_sub_rois)
	sub_rois = [rm.getRoi(i) for i in xrange(n_sub_rois)]

	# Measure
	## Reload unmodified image
	img2 = ImagePlus(image_path+image_fname)
	measures = PA.MEAN \
		+ PA.AREA \
		+ PA.MEDIAN
	rt = ResultsTable()
	ana = Analyzer(img2, measures, rt)

	for i in xrange(n_sub_rois):
		r = rm.getRoi(i)
		img2.setRoi(r)
		ana.measure()

	# Intersect user-defined and detected ROIs
	# Read in user-defined ROIs
	us_rois_fname = image_path + 'RoiSet.zip'
	rm.runCommand('Open', us_rois_fname)

	n_total_rois = rm.getCount()
	n_user_rois = n_total_rois - n_sub_rois

	sub_inds = range(n_sub_rois)
	user_inds = range(n_sub_rois, n_total_rois)

	us_rois = [rm.getRoi(i) for i in user_inds]
	for i in user_inds:
		if re.search(rm.getName(i),image_fname):
			rt.addValue(rm.getName(i), 0.0)
			for j in sub_inds:
				res = str(roi_overlaps(img, rm, i, j))
				print i,j, res
				rt.setValue(rm.getName(i), j, res)

	# WRITE OUT
	rt.saveAs(image_path + image_fname + "_results.txt")


## Variable definition	
image_path = '/home/brad/data/ish/170116_deaf_crhbp/' 

analysis_dir = image_path + 'analysis'
if not os.path.exists(analysis_dir):
	os.makedirs(analysis_dir)
	
image_fnames = os.listdir(image_path)
image_fnames1 = [f for f in image_fnames if re.search("tif$", f) and not re.search("mod", f)]
image_fnames1 = ['or74pu64_crhbp_sec3_L_ra.tif']
for f in image_fnames1:
	analyze_ish_image(image_path, f)