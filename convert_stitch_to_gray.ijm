
source_dir = getDirectory("Source Directory");
//target_dir = getDirectory("Target Directory");
if (File.exists(source_dir)) {
    setBatchMode(true);
    list = getFileList(source_dir);
    for (i=0; i<list.length; i++) {
        if (startsWith(list[i], "Stitch") && !endsWith(list[i], "flat.tif")) {
            open(source_dir + "/" + list[i]);
            run("Stack to RGB");
            run("16-bit");
	    	saveAs("tiff", source_dir + "/" + list[i] + "_flat.tif");
	    	close();
	    	showProgress(i, list.length);
        }
    }
}

